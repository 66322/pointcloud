# Pointcloud research workflow

- User preference (2026-11-01): move directly to experiments. Accept the existing
  82.48% full-val baseline. One frame is enough to check a new executable path;
  do not require 20/100-frame gates or repeat full baseline evaluation.
- Record a commit ID and effective config; no routine SHA/hash scans of files,
  weights, indexes or point arrays. Checksums are for large downloads or an
  actual suspected corruption, not a default experiment step.
- Keep basic shape/label checks, strict checkpoint loading, explicit GPU0/1,
  and separate outputs. These prevent incorrect results without lengthy prep.

- GitHub main is the shared source of truth. Pull with --ff-only before work;
  never force-push. Keep experiment code and small evidence in this repository.
- The user executes server commands manually. Do not claim a server run occurred
  from local tests, reference notes, or a command being prepared.
- Read docs/EXPERIMENT_MASTER.md and docs/RUNBOOK.md before experiment work.
- Server output belongs under /home/jovyan/zzx/. Only GPU 0 and 1 are available
  for this project. Every GPU command must explicitly select those devices.
- Preserve existing DITR checkout, raw data, original/compatible checkpoints,
  baseline predictions and logs. Apply experiments through this repository's
  runtime adapter rather than editing the validated upstream files.
- New model/feature/projection/TTA settings require a fresh output directory.
- Keep machine paths in ignored configs/paths.local.json. Never commit datasets,
  checkpoints, per-point predictions, credentials, or large runtime logs.
- Publish reviewed lightweight outputs under records/experiments/<run_id>/.
  Describe failed results honestly. Separate reported, locally tested,
  server-verified, and scientifically supported claims.
