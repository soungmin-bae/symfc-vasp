# Materials Project reference reconstruction stress test

This validation samples 1,000 crystal structures from Materials Project,
prioritizes structures classified as zero-dimensional molecular crystals, and
tests trajectory-mean reference reconstruction for two lattice transforms:

```text
2 0 0       1 -1 0
0 3 0       0  1 0
0 0 7       1  1 1
```

For each structure, the runner creates 200 synthetic configurations with
independent Cartesian displacements in `[-0.3, 0.3]` angstrom. It then checks
that `symfc-vasp` recovers a strict-symmetry primitive cell, an integer
primitive-to-supercell relation, and a one-to-one species-preserving atom map.

The Materials Project API key is read by `mp-api` from `MP_API_KEY` or the
standard pymatgen configuration. Generated structures and results are kept in
the ignored `artifacts/` directory.

```bash
python validation/mp_reference_stress/stress_test.py prepare --count 1000
python validation/mp_reference_stress/stress_test.py run --workers 8
python validation/mp_reference_stress/stress_test.py summary
```

The run command is resumable. A case is accepted only when every validation
condition is true; exceptions and complete symmetry scans are retained under
`artifacts/failures/`.
