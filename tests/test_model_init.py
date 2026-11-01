import unittest
from types import SimpleNamespace

from pointcloud_lab.model_init import checkpoint_only_dino_init


class ModelInitTests(unittest.TestCase):
    def test_dino_construction_avoids_pretrained_fetch(self):
        calls = []

        def factory(name, pretrained=False, **kwargs):
            if "dinov2" in name and pretrained:
                raise RuntimeError("Would access pretrained cache/network")
            calls.append((name, pretrained, kwargs))
            return object()

        timm = SimpleNamespace(create_model=factory)
        with checkpoint_only_dino_init(timm):
            timm.create_model("vit_large_patch14_reg4_dinov2", pretrained=True,
                              dynamic_img_size=True, num_classes=0)
            timm.create_model("vit_base_patch14_dinov2", True, global_pool="")
            timm.create_model("resnet18", pretrained=True)
        self.assertIs(timm.create_model, factory)
        self.assertFalse(calls[0][1])
        self.assertFalse(calls[1][1])
        self.assertTrue(calls[2][1])
        self.assertTrue(calls[0][2]["dynamic_img_size"])

    def test_strict_load_error_propagates_and_factory_is_restored(self):
        original = lambda *args, **kwargs: object()
        timm = SimpleNamespace(create_model=original)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            with checkpoint_only_dino_init(timm):
                timm.create_model("vit_large_patch14_dinov2", pretrained=True)
                raise RuntimeError("Missing key: backbone.img_enc.vit.weight")
        self.assertIs(timm.create_model, original)


if __name__ == "__main__":
    unittest.main()
