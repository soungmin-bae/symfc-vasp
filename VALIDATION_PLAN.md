# Stabilization acceptance gates

No release is made until every gate below is complete.

## Public contract

- [x] Four-command CLI: `fit`, `phonon`, `gruneisen`, `full`
- [x] Typed Python configs, results, and workflow functions
- [x] FC2 default; FC3 explicit
- [x] Fixed-cell OUTCAR and vasprun.xml parser contract
- [x] General integer 3x3 supercell matrix
- [x] Element and primitive-atom mass overrides

## Scientific diagnostics

- [x] Symmetry candidate scan with integer-cell and atom-map validation
- [x] Fit identifiability and in-sample reconstruction labels
- [x] FC permutation and translational residuals
- [x] Gamma acoustic and imaginary-mode diagnostics
- [x] NAC shape and finite-value validation
- [x] Input hashes, dependency versions, timing, and peak memory
- [x] Vectorized Gruneisen kernels compared with explicit upstream expressions

## Distribution

- [x] Retired sampling and force-collection code removed
- [x] Examples use only supported commands
- [x] Linux/macOS CI and Python 3.11-3.13 matrix defined
- [x] Clean wheel/sdist build and isolated wheel smoke test
- [x] KZP R-3c reference and completed FC2/phonon regression
- [x] CoH3(CN)6 P-31m reference and completed FC2+FC3/Gruneisen regression
- [x] Ohtaka staging-wheel import and CLI smoke test

The package remains version `0.1.1` while the final three external regression
gates are open. Passing them permits consideration of a `0.2.0` alpha release.
