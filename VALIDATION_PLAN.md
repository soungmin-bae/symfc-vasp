# N=3000 validation checklist

- [x] Common trajectory dataset contract
- [x] Streaming OUTCAR parser
- [x] Streaming vasprun.xml parser
- [x] Exact stride-selection unit test
- [x] Minimal vasprun.xml parser test
- [x] Mirror source to Ohtaka
- [x] Inspect the production OUTCAR and confirm 20,000 frames
- [x] Confirm selected indices 5000 through 19995 at stride 5
- [x] Fit FC2 and FC3 on i8cpu
- [x] Generate phonon dispersion
- [x] Generate band-path tensor mode-Gruneisen data
- [x] Generate symmetry-reduced 11x11x11 q-mesh data
- [x] Verify finite arrays, drift, plots, and manifests
- [x] Write `FINAL_VALIDATION.yaml` with `passed: true`

The production `vasprun.xml` contains no usable MD force blocks. Its parser is
validated by unit test; actual OUTCAR/XML numerical equivalence is recorded as
not available rather than inferred.
