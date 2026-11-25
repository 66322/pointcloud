"""Late fusion on image-independent PTv3 features for the proposal experiments."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

BIDIRECTIONAL = {"p_shared", "p_blind", "p_asym", "p_neighbor", "p_prior", "p_distill"}
QUALITY_GATES = BIDIRECTIONAL | {"p_oneway"}
IMAGE_DEPTH = QUALITY_GATES | {"p_imgdepth", "p_reject"}
NEIGHBORS = {"p_neighbor", "p_prior", "p_distill"}
PRIOR = {"p_prior", "p_distill"}


def local_structure(xyz, points=None):
    cells = torch.floor(xyz.float()).long()
    _, index, counts = torch.unique(cells, dim=0, return_inverse=True, return_counts=True)
    n = len(counts)
    count = counts.float()[:, None]
    local = xyz.float() - cells.float()
    mean = local.new_zeros(n, 3).index_add_(0, index, local) / count
    spread = (local - mean[index]).square().sum(1, keepdim=True)
    variance = local.new_zeros(n, 1).index_add_(0, index, spread) / count
    stats = torch.cat((count[index].log1p(), variance[index].sqrt(),
                       spread.sqrt(), xyz.float().norm(dim=1, keepdim=True) / 50), dim=1)
    if points is None:
        return stats, None
    values = points.float()
    pooled = values.new_zeros(n, values.shape[1]).index_add_(0, index, values) / count
    maximum = values.new_full(pooled.shape, torch.finfo(values.dtype).min)
    maximum = maximum.scatter_reduce(0, index[:, None].expand_as(values), values, reduce="amax")
    return stats, torch.cat((points, pooled[index], maximum[index]), dim=1)


def image_statistics(normalized, patch):
    rgb = (normalized.float() * normalized.new_tensor([.229, .224, .225])[:, None, None]
           + normalized.new_tensor([.485, .456, .406])[:, None, None]).clamp(0, 1)
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[:, None, None]).sum(0, keepdim=True)
    padded = F.pad(gray[None], (1, 1, 1, 1), mode="replicate")[0]
    dx = padded[:, 1:-1, 2:] - gray
    dy = padded[:, 2:, 1:-1] - gray
    lap = dx + dy + padded[:, 1:-1, :-2] + padded[:, :-2, 1:-1] - 2 * gray
    moments = F.avg_pool2d(torch.cat((rgb, gray, gray.square(), dx.abs() + dy.abs(), lap.abs())), patch)
    quality = torch.cat((moments[3:4], (moments[4:5] - moments[3:4].square()).clamp_min(0).sqrt(),
                         moments[5:7]))
    return torch.cat((moments[:3], quality)), quality


def grid_values(grid, xy):
    # xy is in token-center coordinates. Accumulate and interpolate in fp32 under AMP.
    _, height, width = grid.shape
    unit = 2 * (xy.float() + .5) / xy.new_tensor([width, height]).float() - 1
    return F.grid_sample(grid[None].float(), unit[None], padding_mode="border",
                         align_corners=False)[0].permute(1, 2, 0)


def depth_buffer(xy, z, valid, h, w):
    pix = (xy[valid] + .5).floor().long()
    ids = pix[:, 1].clamp(0, h - 1) * w + pix[:, 0].clamp(0, w - 1)
    return z.new_full((h * w,), torch.inf).scatter_reduce(0, ids, z[valid], reduce="amin")


def view_selection(uv, valid, keep, primary, pose, h, w, allow_neighbors):
    rows = torch.arange(len(primary), device=primary.device)
    available = valid & keep
    current = available[rows, primary]
    adjacent = available & pose[primary, :, 0].bool()
    neighbor = ~current & adjacent.any(1) & allow_neighbors
    cost = ((uv + .5) / uv.new_tensor([w, h]) - .5).square().sum(-1)
    nearest = cost.masked_fill(~adjacent, torch.inf).argmin(1)
    camera = torch.where(current, primary, torch.where(neighbor, nearest, -1))
    return camera, neighbor


class ProposalFusion(nn.Module):
    def __init__(self, method, image_channels, width, hidden):
        super().__init__()
        self.method = method
        self.visual = nn.Conv2d(image_channels, width, 1)
        self.detail = nn.Conv2d(7, width, 1, bias=False)
        self.query = nn.Linear(width, width)
        self.offset = nn.Linear(width, 10)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.geometry = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.image_depth = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.image_depth.weight)
        nn.init.constant_(self.image_depth.bias, 3.)
        self.null = nn.Linear(width, 1)
        self.gates = nn.Sequential(nn.Linear(2 * width + 9, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.to_image = nn.Linear(width, width, bias=False)
        self.inject = nn.Linear(width, width, bias=False)
        self.pose_gate = nn.Linear(3, 1)
        self.prior = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
        self.register_buffer("anchors", torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -2], [0, 2.]]))
        self.totals = None
        active = {"visual", "detail", "inject"} if method != "p_lidar" else set()
        if method not in {"p_lidar", "p_center"}:
            active |= {"query", "offset"}
        if method not in {"p_lidar", "p_center", "p_offset"}:
            active.add("geometry")
        if method in IMAGE_DEPTH:
            active.add("image_depth")
        if method in QUALITY_GATES or method == "p_reject":
            active.add("null")
        if method in QUALITY_GATES:
            active.add("gates")
        if method in BIDIRECTIONAL:
            active.add("to_image")
        if method in NEIGHBORS:
            active.add("pose_gate")
        if method in PRIOR:
            active.add("prior")
        for name, module in self.named_children():
            module.requires_grad_(name in active)

    def candidates(self, p, grid, xy, z, depths, neighbor, predicted_depth):
        h, w = grid.shape[-2:]
        if self.method == "p_center":
            samples = xy[:, None]
            weights = p.new_ones(len(p), 1).float()
            return samples, weights, grid_values(grid, samples)[:, 0]
        delta = self.offset(p).float().reshape(-1, 5, 2).tanh()
        # Preserve the center as a valid candidate even near an image boundary.
        delta = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, 1:]), dim=1)
        samples = xy[:, None] + self.anchors + delta
        pix = (samples + .5).floor().long()
        valid = ((pix[..., 0] >= 0) & (pix[..., 0] < w)
                 & (pix[..., 1] >= 0) & (pix[..., 1] < h))
        # Neighbor compensation only considers center and the two short-range candidates.
        valid[:, 3:] &= ~neighbor[:, None]
        values = grid_values(grid, samples)
        score = (F.normalize(self.query(p).float(), dim=-1)[:, None]
                 * F.normalize(values, dim=-1)).sum(-1) * 5
        if self.method != "p_offset":
            sample_depth = depths[pix[..., 1].clamp(0, h - 1) * w + pix[..., 0].clamp(0, w - 1)]
            known = sample_depth.isfinite()
            diff = (z[:, None] - torch.where(known, sample_depth, z[:, None])).abs()
            rel = (diff / (.5 + .03 * z[:, None])).clamp_max(10)
            geom = torch.stack((known.float(), rel, (samples - xy[:, None]).norm(dim=-1)), -1)
            score = score + self.geometry(geom).squeeze(-1).float() - rel
        score = score.masked_fill(~valid, -torch.inf)
        if predicted_depth is not None:
            image_log_z = grid_values(predicted_depth, samples).squeeze(-1)
            score = score - (image_log_z - z.clamp_min(.1).log()[:, None]).abs() / .25
        if self.method in QUALITY_GATES or self.method == "p_reject":
            score = torch.cat((score, self.null(p).float()), dim=1)
        weights = score.softmax(-1)[:, :5]
        return samples, weights, (values * weights[..., None]).sum(1)

    def frame(self, p, xyz, coords, masks, z, primary, keep, pose, maps, rgb, patch, supervise_depth):
        h, w = maps.shape[-2:]
        xy = (coords.float() + .5) / patch - .5
        valid = (masks.bool() & torch.isfinite(xy).all(-1) & (z > 0)
                 & (coords[..., 0] >= 0) & (coords[..., 0] < w * patch)
                 & (coords[..., 1] >= 0) & (coords[..., 1] < h * patch))
        xy = torch.stack((xy[..., 0].clamp(0, w - 1), xy[..., 1].clamp(0, h - 1)), dim=-1)
        owner, neighbor = view_selection(coords, valid, keep, primary, pose, h * patch,
                                         w * patch, self.method in NEIGHBORS)
        q3, pooled = (local_structure(xyz, p if self.method in PRIOR else None)
                      if self.method in QUALITY_GATES else (None, None))
        fallback = owner < 0
        result = p.clone()
        if pooled is not None:
            result[fallback] = p[fallback] + self.prior(pooled[fallback]).to(p.dtype)
        counters = [p.new_zeros((), dtype=torch.float32) for _ in range(3)]
        depth_losses = []
        for camera, imap in enumerate(maps):
            idx = torch.where(owner == camera)[0]
            if not len(idx):
                continue
            detail, q2 = image_statistics(rgb[camera], patch)
            grid = self.visual(imap) + self.detail(detail)
            center = xy[idx, camera]
            depth = depth_buffer(xy[:, camera], z[:, camera], valid[:, camera], h, w)
            predicted_depth = self.image_depth(grid) if self.method in IMAGE_DEPTH else None
            if supervise_depth and predicted_depth is not None:
                known = depth.isfinite()
                depth_losses.append(F.smooth_l1_loss(
                    predicted_depth.flatten()[known].float(), depth[known].clamp_min(.1).log()))
            samples, weights, visual = self.candidates(
                p[idx], grid, center, z[idx, camera], depth, neighbor[idx], predicted_depth)
            mass = weights.sum(-1, keepdim=True)
            if self.method in QUALITY_GATES:
                qi = grid_values(q2, center[:, None])[:, 0]
                cues = torch.cat((qi, q3[idx]), dim=1)
                if self.method == "p_blind":
                    cues = torch.zeros_like(cues)
                gates = self.gates(torch.cat((p[idx], visual, cues, mass), dim=1)).sigmoid()
                if self.method == "p_shared":
                    gates = gates.mean(1, keepdim=True).expand(-1, 2)
                incoming, outgoing = gates.float().split(1, dim=1)
            if self.method in BIDIRECTIONAL:
                pix = (center + .5).floor().long()
                cells = pix[:, 1].clamp(0, h - 1) * w + pix[:, 0].clamp(0, w - 1)
                values = self.to_image(p[idx]).float() * outgoing * mass
                summed = values.new_zeros(h * w, values.shape[1]).index_add(0, cells, values)
                counts = torch.bincount(cells, minlength=h * w).clamp_min(1)
                message = (summed / counts[:, None]).T.reshape(-1, h, w)
                visual = visual + (grid_values(message, samples) * weights[..., None]).sum(1)
                counters[2] = counters[2] + outgoing.detach().sum()
            if self.method in QUALITY_GATES:
                visual = visual * incoming
                counters[1] = counters[1] + incoming.detach().sum()
            if self.method in NEIGHBORS:
                pose_weight = self.pose_gate(pose[primary[idx], camera, 1:]).sigmoid()
                visual = visual * torch.where(neighbor[idx, None], pose_weight, 1.)
            result[idx] = p[idx] + self.inject(visual).to(p.dtype)
            counters[0] = counters[0] + mass.detach().sum()
        if not self.training:
            stats = torch.stack([(~fallback).sum(), neighbor.sum(), fallback.sum(), *counters]).detach().double()
            self.totals = stats if self.totals is None else self.totals + stats
        return result, fallback, depth_losses

    def forward(self, p, data, maps, patch, full=False):
        starts = [0, *data.get("unmix3d_offset", data["offset"]).tolist()]
        if len(starts) != len(maps) + 1 or starts[-1] != len(p):
            raise ValueError("Source frame offsets do not match image features")
        features, missing, losses = [], [], []
        for b, (start, end) in enumerate(zip(starts, starts[1:])):
            s = slice(start, end)
            keep = data["lab_keep"][b]
            out, mask, depth_losses = self.frame(p[s], data["coord"][s], data["image_coord"][s], data["image_mask"][s],
                                   data["lab_depth"][s], data["lab_primary"][s], torch.ones_like(keep)
                                   if full else keep, data["lab_pose"][b], maps[b], data["image"][b], patch,
                                   self.training and not full and bool(data["lab_depth_ok"][b]))
            features.append(out)
            missing.append(mask)
            losses.extend(depth_losses)
        return torch.cat(features), torch.cat(missing), torch.stack(losses).mean() if losses else None

    def diagnostics(self):
        if self.totals is None:
            return {}
        observed, neighbor, fallback, mass, incoming, outgoing = self.totals.tolist()
        return {"unit": "point visits across TTA fragments", "direct_visits": int(observed - neighbor),
                "neighbor_visits": int(neighbor), "fallback_visits": int(fallback),
                "mean_visual_mass": mass / observed if observed else None,
                "mean_image_to_point": incoming / observed if observed and self.method in QUALITY_GATES else None,
                "mean_point_to_image": outgoing / observed if observed and self.method in BIDIRECTIONAL else None}


def point_only_unpool(self, point):
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    coarse = self.proj(point)
    parent = self.proj_skip(parent)
    parent.feat = parent.feat + coarse.feat[inverse]
    if self.traceable:
        parent["unpooling_parent"] = coarse
    return parent


def encode_views(encoder, extract, data, full_teacher):
    image, keep = data["image"], data["lab_keep"].bool()
    if full_teacher or keep.all():
        maps, _, patch = extract(encoder, data)
        return maps, patch
    b, v, _, h, w = image.shape
    indices = torch.where(keep.flatten())[0]
    if not len(indices):
        patch = encoder.patch_size
        return image.new_zeros(()).expand(b, v, encoder.output_channels, h // patch, w // patch), patch
    # Encode only actual student observations. Every DINO input is an independent image.
    selected = image.flatten(0, 1).index_select(0, indices)
    subset, _, patch = extract(encoder, {"image": selected[None]})
    restored = subset.new_zeros(b * v, *subset.shape[2:])
    restored.index_copy_(0, indices, subset[0])
    return restored.reshape(b, v, *subset.shape[2:]), patch


def install_proposal(model, image_module, method, protocol):
    backbone = model.backbone
    fusion = ProposalFusion(method, backbone.img_enc.output_channels, model.seg_head.in_features,
                            protocol["gate_hidden"])
    model.add_module("lab_fusion", fusion)
    count = 0
    for module in backbone.modules():
        if isinstance(module, image_module.SerializedUnpooling):
            module.proj_image_feat.requires_grad_(False)
            module.forward = MethodType(point_only_unpool, module)
            count += 1
    if count != 4:
        raise ValueError(f"Expected four DITR decoder stages, found {count}")

    def forward(self, data):
        point = image_module.Point({k: data[k] for k in ("coord", "grid_coord", "offset", "feat")})
        point.serialization(order=backbone.order, shuffle_orders=backbone.shuffle_orders)
        point.sparsify()
        base = backbone.dec(backbone.enc(backbone.embedding(point))).feat
        aux = None
        depth_loss = None
        if method == "p_lidar":
            logits = self.seg_head(base)
        else:
            maps, patch = encode_views(backbone.img_enc, image_module.get_image_feat, data,
                                       self.training and method == "p_distill")
            features, fallback, depth_loss = fusion(base, data, maps, patch)
            logits = self.seg_head(features)
            if self.training and method == "p_distill":
                eligible = fallback & data["image_mask"].any(1) & (data["segment"] >= 0)
                if eligible.any():
                    with torch.no_grad():
                        teacher, _, _ = fusion(base.detach(), data, maps, patch, full=True)
                        target = (self.seg_head(teacher[eligible]).float() / 2).softmax(-1)
                    aux = 4 * F.kl_div((logits[eligible].float() / 2).log_softmax(-1),
                                       target, reduction="batchmean")
        result = {"seg_logits": logits}
        if "segment" in data:
            loss = self.criteria(logits, data["segment"])
            if depth_loss is not None:
                loss = loss + .1 * depth_loss
            result["loss"] = loss if aux is None else loss + .5 * aux
        return result

    model.forward = MethodType(forward, model)
