# Pointcloud research workflow

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
