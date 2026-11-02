# Dataset provenance and access

**Source:** Design-Space Dimensionality Reduction Benchmark Dataset – DTMB 5415 Hull, Serani and Diez, Zenodo.

- DOI: https://doi.org/10.5281/zenodo.19222008
- Original dataset documentation and matrix metadata are preserved in `source_documentation/`.
- Obtain the original archive from the publisher; **the large `.mat` file is not included here**.
- Expected local filename for the scripts: `DTMB5415-database.mat` in the repository root. The source documentation also names `DTMB5415_database.mat`; rename your local downloaded copy to match the script if necessary.
- The source licensing terms: Creative Commons Attribution 4.0 International


## Input and target mapping

`variablesMatrix`: 27 × 9001, one geometry per column. `coefStatMatrix`: 3 × 9001, Cw for Fr 0.25, 0.33 and 0.41 by row. Each geometry creates three observations; inputs comprise 27 geometry parameters plus Fr. The supervised-learning target is a numerical wave-resistance coefficient, not total drag or experimentally validated hydrodynamic resistance. Labels come from WARP linear potential-flow calculations.

Hull coordinates, pressures and free-surface elevations are not included as ML features. The 27 original shape parameters have no individually confirmed physical names in supplied metadata.

The authors' dataset README says configurations were physically valid under the original screening procedure; that does not prove every new generated combination is physically admissible. Some supplied coefficients are negative, and outputs should not be clipped without an explicit methodological justification.
