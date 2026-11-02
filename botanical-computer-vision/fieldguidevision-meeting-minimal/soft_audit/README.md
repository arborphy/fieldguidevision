# Soft-clustering audit and rerun

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python botanical-computer-vision/fieldguidevision-meeting-minimal/soft_audit/run_soft_clustering.py`, then `python botanical-computer-vision/fieldguidevision-meeting-minimal/soft_audit/update_report.py`. Run the K report updater first if regenerating both. NumPy, pandas, SciPy, scikit-learn, matplotlib and threadpoolctl are required.

The original script is archived in `../changed_files.zip` as `botanical-computer-vision/meeting_minimal/exp4_soft.py`.

## Findings in the old implementation

- Responsibilities from `GaussianMixture.predict_proba` are mathematically valid posterior allocations among mutually exclusive mixture components. They are not probabilities that multiple organs coexist in an image. A diagonal GMM becoming nearly hard is not by itself an arithmetic bug or proof that images lack mixed organs.
- The error AUROC used `all_predictions.csv`, whose columns are `predicted_species`, `ground_truth_species`, and `correct`. Thus the earlier approximately 0.51 value measured species-classification error, not organ-classification error. The report's general conclusion about error detection was too broad.
- PCA and GMM were fit on all 1,641 images, including the reported test subset. The old report called the fit descriptive, but its error-detection number is not an inductive held-out validation of that fitted GMM. The redo fits train only and uses validation; no new test diagnostics are computed.
- BIC considered multiple K but the displayed memberships always used K=20. The redo selects one density model by training BIC within a fixed 32-D space, while showing the full sweep. That density choice is not a biological optimum.
- Raw entropy has a K-dependent upper bound. The new plots use H/log(K), report multiple pmax thresholds, and keep the arbitrary threshold 0.6 explicitly labelled.
- The checked-in 180-row multi-label audit CSV is an empty annotation template: every `visible_*`, `ambiguous`, and `acceptable_labels` cell is blank. It supplies no multi-organ ground truth.

## New protocol

PCA is fit on the 1,181 training images only, without whitening. GMM candidates use fixed PCA-32, K=5/10/15/20/30/50, diagonal/tied/full covariances, seeds 42–46, three initializations per seed, covariance regularization 1e-6, and 300 maximum EM iterations. Five-seed variation measures optimization sensitivity, not population uncertainty. Full covariance at large K is deliberately included as an overfitting diagnostic, not assumed preferable.

A separate dimension-sensitivity experiment uses PCA-8/16, K=5/10/20/50, diagonal/tied covariance, and three seeds. BIC and log density are only compared for models in the same dimension and coordinate transformation; their absolute values are not comparable across different PCA dimensions.

The selected model is the converged 32-D candidate with minimum training BIC (including seed selection), with no validation/test tuning. All memberships, densities, validation errors and bootstrap AUROCs refer to that frozen candidate. Other fits are retained in `runs.csv` and `summary.csv`. Normalized entropy and pmax are model-conditional allocation scores; they are uncalibrated for organ presence or correctness. Even a low-density outlier can have pmax near one.

The organ-error reference is a fixed C=10, balanced logistic regression trained on raw EfficientNet vectors, excluding canonical `other` labels. This reference is not the old species classifier. Its own validation macro-F1 and errors are reported. AUROC confidence intervals resample whole validation individual plants (500 draws), conditional on the selected density model and trained probe. They exclude model-training uncertainty and the uncertainty from choosing a model. Shot scale is provisional; multi-organ accuracy is not evaluated without filled annotations.

`selected_responsibilities.npz` contains all K responsibility columns on validation. Train component organ composition is computed by responsibility-weighted counts; component IDs are not organ labels. Ambiguous examples are ranked by mean normalized entropy across seeds for the selected K/covariance; their detailed probabilities use the train-BIC selected seed. No per-component cross-seed alignment is assumed. Seed stability uses permutation-invariant ARI, entropy rank correlation and top-25-set Jaccard.

The old all-corpus PCA32/diag20 configuration is rerun as a historical diagnostic using the installed scikit-learn 1.8.0 and float64. The archived report names scikit-learn 1.9 and the old loader casts embeddings to float32, so these numbers are not claimed to reproduce its exact values. The same input embedding SHA-256 is checked. Near-hard membership persists in the old configuration and in the corrected held-out fits.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
