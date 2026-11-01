"""Exercise actual pinned upstream pool/unpool classes with a CPU CSR stand-in.

The stand-in replaces the unavailable CUDA kernel only. The upstream forward
methods are extracted unchanged, so traceable parents, RNG, and support mapping
are checked against the real source rather than a second adapter implementation.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/ditr_image_contract/point_transformer_v3m1_image.py"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def upstream_classes():
    import torch
    from torch import nn
    class Point(dict):
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def sparsify(self):
            return self
    class Sequential(nn.Sequential):
        def add(self, module):
            self.add_module(str(len(self)), module)
        def forward(self, value):
            if isinstance(value, Point):
                value.feat = super().forward(value.feat)
                return value
            return super().forward(value)
    def csr(source, pointer, reduce):
        result = []
        for start, end in zip(pointer[:-1].tolist(), pointer[1:].tolist()):
            part = source[start:end]
            if reduce == "max":
                result.append(part.max(0).values)
            elif reduce == "mean":
                result.append(part.mean(0))
            else:
                raise ValueError(reduce)
        return torch.stack(result)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"SerializedPooling", "SerializedUnpooling"}]
    namespace = dict(torch=torch, nn=nn, torch_scatter=SimpleNamespace(segment_csr=csr),
                     Point=Point, Dict=Point, PointModule=nn.Module, PointSequential=Sequential)
    import math
    namespace["math"] = math
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


@unittest.skipUnless(HAS_TORCH and SOURCE.is_file(), "Needs Torch and cached pinned upstream source")
class ValidityTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn
        self.api = upstream_classes()
        self.Point = self.api["Point"]
        self.model = nn.Module()
        self.model.backbone = nn.Module()
        self.model.backbone.img_enc = nn.Linear(1, 1)
        self.model.backbone.pools = nn.ModuleList([
            self.api["SerializedPooling"](3, 3, shuffle_orders=True,
                                          norm_layer=nn.BatchNorm1d, act_layer=nn.GELU)
            for _ in range(4)])
        self.model.backbone.ups = nn.ModuleList([
            self.api["SerializedUnpooling"](3, 3, 2, 3) for _ in range(4)])
        self.module = SimpleNamespace(
            SerializedPooling=self.api["SerializedPooling"],
            SerializedUnpooling=self.api["SerializedUnpooling"],
            assign_image_feat=lambda *args: None)
        self.model.eval()

    def inputs(self, all_valid=False):
        import torch
        return self.Point(
            feat=torch.arange(18.).reshape(6, 3), coord=torch.arange(18.).reshape(6, 3),
            image_feat=torch.tensor([[-2., -3.], [0., 0.], [0., 0.], [0., 0.], [-8., -5.], [-6., -7.]]),
            lab_image_valid=torch.ones(6, dtype=torch.bool) if all_valid else
                torch.tensor([True, False, False, False, True, True]),
            grid_coord=torch.arange(18).reshape(6, 3), batch=torch.zeros(6, dtype=torch.long),
            serialized_code=torch.tensor([[0, 0, 8, 8, 16, 16], [0, 0, 8, 8, 16, 16]]),
            serialized_order=torch.arange(6).repeat(2, 1),
            serialized_inverse=torch.arange(6).repeat(2, 1), serialized_depth=3,
        )

    def install(self, method):
        from pointcloud_lab.training_models import install_treatment
        install_treatment(self.model, self.module, method,
                          {"seed": 42, "gate_hidden": 4, "camera_dropout": .2})

    def test_masked_max_preserves_negative_observations_and_restores_skip_parent(self):
        import torch
        self.install("valid_pool")
        point = self.inputs()
        original_image = point.image_feat.clone()
        result = self.model.backbone.pools[0](point)
        torch.testing.assert_close(result.image_feat, torch.tensor([[-2., -3.], [0., 0.], [-6., -5.]]))
        self.assertTrue(torch.equal(result["lab_image_valid"], torch.tensor([True, False, True])))
        torch.testing.assert_close(result.pooling_parent.image_feat, original_image, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result.image_feat).all())

    def test_all_valid_is_exactly_original_and_pooling_rng_unchanged(self):
        import torch
        pool = self.model.backbone.pools[0]
        torch.manual_seed(43)
        expected = pool(self.inputs(all_valid=True))
        rng = torch.get_rng_state()
        self.install("valid_both")
        torch.manual_seed(43)
        actual = pool(self.inputs(all_valid=True))
        torch.testing.assert_close(actual.feat, expected.feat, rtol=0, atol=0)
        torch.testing.assert_close(actual.image_feat, expected.image_feat, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_post_projection_mask_removes_bias_and_multi_level_support_survives(self):
        import torch
        self.install("valid_both")
        point = self.inputs()
        coarse = self.model.backbone.pools[0](point)
        deeper = self.model.backbone.pools[1](coarse)
        recovered = self.model.backbone.ups[1](deeper)
        self.assertTrue(torch.equal(recovered["lab_image_valid"], torch.tensor([True, False, True])))
        recovered = self.model.backbone.ups[0](recovered)
        self.assertTrue(torch.equal(recovered["lab_image_valid"], point["lab_image_valid"]))
        self.assertTrue(torch.isfinite(recovered.feat).all())
        # Direct injection test: nonzero projection bias must be removed only
        # where the skip parent's observation support is false.
        up = self.model.backbone.ups[0]
        parent = self.Point(feat=torch.zeros(2, 3), image_feat=torch.zeros(2, 2),
                            lab_image_valid=torch.tensor([True, False]))
        with torch.no_grad():
            for layer in (up.proj[0], up.proj_skip[0]):
                layer.weight.zero_()
                layer.bias.zero_()
            up.proj_image_feat[1].weight.zero_()
            up.proj_image_feat[1].bias.fill_(3.)
        result = up(self.Point(feat=torch.zeros(1, 3), pooling_parent=parent,
                               pooling_inverse=torch.zeros(2, dtype=torch.long)))
        torch.testing.assert_close(result.feat, torch.tensor([[3., 3., 3.], [0., 0., 0.]]))


if __name__ == "__main__":
    unittest.main()
