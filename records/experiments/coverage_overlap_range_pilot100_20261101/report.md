# Existing-cache diagnostic

Frames: 100; labelled points: 2496549.
Fixed-16 mIoU: 84.3997%; present-union classes: 16/16.
A pilot is not a full validation result. No improved-model claim.

| Group | Points (incl ignored) | Labelled | mIoU fixed16 % | OA % |
|---|---:|---:|---:|---:|
| all | 3471360 | 2496549 | 84.3997 | 94.70637267684312 |
| camera_0 | 1542881 | 575794 | 55.6502 | 95.31933295588352 |
| camera_1 | 1763251 | 1756153 | 84.4956 | 94.57888919701188 |
| camera_ge2 | 165228 | 164602 | 79.8911 | 93.92230957096511 |
| range_0_20m | 2983654 | 2010839 | 87.2538 | 94.80873406573077 |
| range_20_40m | 338029 | 336572 | 75.1798 | 94.44665628751055 |
| range_ge40m | 149677 | 149138 | 43.4535 | 93.91234963590769 |

Frustum coverage is not physical visibility.
Per-class absent unions are null in JSON/CSV; fixed16 matches upstream zero convention.
Distance × class counts/IoUs are in per_class.csv for each range group.
