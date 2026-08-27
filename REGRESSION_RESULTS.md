# Real-case regression results

Validated on 2026-08-27 with the local `0.1.1` stabilization branch.
Machine-specific source paths are intentionally omitted; trajectory SHA256 is
the portable identity.

| Case | Trajectory SHA256 | Frames | Reference | Supercell relation | Existing numerical products |
|---|---|---:|---|---|---|
| KZP | `2d9eed98aa03c8067879e2fb1a5b93d029fd82e90f49b7150ed52c6a397db013` | 200 | R-3c (167), 36 atoms | 2x2x2, 288 atoms | FC2 and phonon band |
| CoH3(CN)6 | `379a74e3ce6961262b074cba9fc21673dd1a5bc79e2b4ab37b2b9ba049afb717` | 500 | P-31m (162), 16 atoms | 2x2x2, 128 atoms | FC2, FC3, phonon band, 11x11x11 Gruneisen mesh |

The revised reference selector was rerun independently from both trajectories.
It reproduced R-3c at `symprec=0.15 A` with maximum mapping residual
`0.0785409 A`, and P-31m at `symprec=0.03 A` with maximum mapping residual
`0.017553 A`. Both reconstructed the integer diagonal 2x2x2 relation without
using the completed fitting outputs.

The CoH3(CN)6 completed result contains a `[146, 48, 3, 3]` irreducible-mesh
tensor and 12 band segments with 21 points per segment. The KZP completed band
has only numerical Gamma-scale round-off below zero in its accepted run.
