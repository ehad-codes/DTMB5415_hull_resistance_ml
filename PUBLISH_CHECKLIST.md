# Publication review checklist

- [x] Reviewed all supplied Python scripts for core dataset/prediction paths.
- [x] Re-ran saved-model evaluation and hull ranking in a separate environment; headline metrics matched.
- [x] Identified and corrected a display-only best-epoch reporting issue in `compare_mlp_architectures.py` (does not modify saved model weights or reported performance).
- [x] Confirm dataset source licence permits copying original README/metadata and sharing processed arrays. If not, remove `source_documentation/` and/or `processed_data/` from public upload and replace with source links.
- [x] Verify the exact original archive filename and license statement directly on publisher page.
- [x] Test `pip install -r requirements.txt` in clean Python 3.14.6 environment (not tested in audit).
- [ ] Run evaluation and plotting scripts in the user's final environment.
- [ ] Check README figures visually and check plot axes/legibility.
- [ ] Check public Git history excludes raw 1.09 GB `.mat` and `.venv`.
- [ ] Decide whether original project plan belongs in repository; its unfinished checklist is historical.
- [ ] Do not present execution speed-up over numerical solver as measured; none was benchmarked.
- [ ] Do not assert numerical generation, true optimiser, RANS or experimental validation.

**Audit note:** Tested evaluation with Python 3.13 and a different installed library stack: sklearn 1.8.0 raised version-warning when reading saved sklearn 1.9.1 joblib objects, but metrics matched supplied CSV. Avoid treating this as a clean-environment reproduction test.
