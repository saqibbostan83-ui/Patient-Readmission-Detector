# Final Research Replication Report

Generated: 2026-11-22 15:03:43

## 1. Research objective
Replication-oriented evaluation of machine learning models for predicting 30-day hospital readmission among patients with diabetes using the UCI Diabetes 130-US Hospitals dataset.

## 2. Models
Logistic Regression, Random Forest, XGBoost, and Deep Neural Network.

## 3. Experiment verification
| protocol   |   models_in_leaderboard |   models_with_saved_predictions |   test_encounters |   patient_overlap |   features | verified   |
|:-----------|------------------------:|--------------------------------:|------------------:|------------------:|-----------:|:-----------|
| encounter  |                       4 |                               4 |             19699 |              8223 |        261 | True       |
| patient    |                       4 |                               4 |             19608 |                 0 |        261 | True       |

## 4. Paper versus project ROC-AUC
| protocol   | model   |   paper_roc_auc |   our_test_roc_auc |   difference |   our_test_pr_auc |   our_brier |
|:-----------|:--------|----------------:|-------------------:|-------------:|------------------:|------------:|
| encounter  | A_LR    |          0.6420 |             0.6682 |       0.0262 |            0.2214 |      0.2272 |
| encounter  | A_RF    |          0.6300 |             0.6798 |       0.0498 |            0.2318 |      0.2002 |
| encounter  | A_XGB   |          0.6670 |             0.6778 |       0.0108 |            0.2369 |      0.2215 |
| encounter  | A_DNN   |          0.5790 |             0.6749 |       0.0959 |            0.2296 |      0.2459 |
| patient    | A_LR    |          0.6420 |             0.6651 |       0.0231 |            0.2149 |      0.2256 |
| patient    | A_RF    |          0.6300 |             0.6724 |       0.0424 |            0.2168 |      0.1705 |
| patient    | A_XGB   |          0.6670 |             0.6754 |       0.0084 |            0.2271 |      0.2207 |
| patient    | A_DNN   |          0.5790 |             0.6698 |       0.0908 |            0.2194 |      0.2270 |

## 5. Encounter-level results
| protocol   | leaderboard   | experiment   |   cv_roc_auc |   cv_roc_auc_std |   cv_pr_auc |   test_roc_auc |   test_roc_auc_ci_low |   test_roc_auc_ci_high |   test_pr_auc |    brier |   acc@0.5 |   prec@0.5 |   rec@0.5 |   f1@0.5 |   spec@0.5 |   thr_oof_f1 |   acc@thr |   prec@thr |   rec@thr |   f1@thr |   spec@thr |   tn@thr |   fp@thr |   fn@thr |   tp@thr |   train_auc_refit |   n_features |   n_test | best_params                                                  |   seconds |
|:-----------|:--------------|:-------------|-------------:|-----------------:|------------:|---------------:|----------------------:|-----------------------:|--------------:|---------:|----------:|-----------:|----------:|---------:|-----------:|-------------:|----------:|-----------:|----------:|---------:|-----------:|---------:|---------:|---------:|---------:|------------------:|-------------:|---------:|:-------------------------------------------------------------|----------:|
| encounter  | A_replication | A_LR         |     0.662531 |       0.00690629 |    0.213163 |       0.668165 |              0.655818 |               0.680418 |      0.221387 | 0.227229 |  0.662318 |   0.181119 |  0.554126 | 0.273005 |   0.676297 |     0.52367  |  0.705873 |   0.192602 |  0.492014 | 0.276835 |   0.733505 |    12796 |     4649 |     1145 |     1109 |          0.67115  |          261 |    19699 | {"C": 0.01}                                                  |   85.4206 |
| encounter  | A_replication | A_RF         |     0.67085  |       0.00549455 |    0.222821 |       0.67978  |              0.666451 |               0.692228 |      0.231766 | 0.200207 |  0.724859 |   0.206635 |  0.494676 | 0.291503 |   0.7546   |     0.514887 |  0.754962 |   0.218305 |  0.442325 | 0.292333 |   0.795357 |    13875 |     3570 |     1257 |      997 |          0.906161 |          261 |    19699 | {"max_depth": null, "min_samples_leaf": 10}                  |  791.686  |
| encounter  | A_replication | A_XGB        |     0.669786 |       0.00629521 |    0.225597 |       0.677824 |              0.665112 |               0.69073  |      0.23692  | 0.221546 |  0.653739 |   0.185685 |  0.598492 | 0.283433 |   0.660877 |     0.542694 |  0.731509 |   0.208341 |  0.480923 | 0.290734 |   0.763887 |    13326 |     4119 |     1170 |     1084 |          0.700972 |          261 |    19699 | {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 500} |  250.455  |
| encounter  | A_replication | A_DNN        |     0.666495 |       0.00499469 |    0.222129 |       0.674928 |              0.66277  |               0.687301 |      0.229639 | 0.245899 |  0.561551 |   0.163592 |  0.688554 | 0.264373 |   0.545142 |     0.589125 |  0.752678 |   0.212813 |  0.430346 | 0.284792 |   0.794325 |    13857 |     3588 |     1284 |      970 |          0.690568 |          261 |    19699 | {"dropout": 0.4, "hidden": [128, 64]}                        |  272.914  |

## 6. Patient-level results
| protocol   | leaderboard   | experiment   |   cv_roc_auc |   cv_roc_auc_std |   cv_pr_auc |   test_roc_auc |   test_roc_auc_ci_low |   test_roc_auc_ci_high |   test_pr_auc |    brier |   acc@0.5 |   prec@0.5 |   rec@0.5 |   f1@0.5 |   spec@0.5 |   thr_oof_f1 |   acc@thr |   prec@thr |   rec@thr |   f1@thr |   spec@thr |   tn@thr |   fp@thr |   fn@thr |   tp@thr |   train_auc_refit |   n_features |   n_test | best_params                                                 |   seconds |
|:-----------|:--------------|:-------------|-------------:|-----------------:|------------:|---------------:|----------------------:|-----------------------:|--------------:|---------:|----------:|-----------:|----------:|---------:|-----------:|-------------:|----------:|-----------:|----------:|---------:|-----------:|---------:|---------:|---------:|---------:|------------------:|-------------:|---------:|:------------------------------------------------------------|----------:|
| patient    | A_replication | A_LR         |     0.663261 |       0.00419332 |    0.214522 |       0.665135 |              0.65295  |               0.67787  |      0.214939 | 0.225581 |  0.669676 |   0.183052 |  0.550718 | 0.274773 |   0.684925 |     0.520777 |  0.708741 |   0.193128 |  0.491921 | 0.277363 |   0.736536 |    12801 |     4579 |     1132 |     1096 |          0.67197  |          261 |    19608 | {"C": 0.01}                                                 |   168.468 |
| patient    | A_replication | A_RF         |     0.672465 |       0.00383902 |    0.220393 |       0.67239  |              0.660185 |               0.684008 |      0.216844 | 0.170457 |  0.812169 |   0.242478 |  0.307451 | 0.271126 |   0.87687  |     0.450637 |  0.733629 |   0.20411  |  0.463645 | 0.283441 |   0.768239 |    13352 |     4028 |     1195 |     1033 |          0.930733 |          261 |    19608 | {"max_depth": null, "min_samples_leaf": 10}                 |  2197.09  |
| patient    | A_replication | A_XGB        |     0.670302 |       0.00386387 |    0.226563 |       0.675409 |              0.663048 |               0.688127 |      0.227127 | 0.22065  |  0.65917  |   0.185515 |  0.589767 | 0.282247 |   0.668067 |     0.563025 |  0.76734  |   0.217843 |  0.404399 | 0.283155 |   0.813867 |    14145 |     3235 |     1327 |      901 |          0.69855  |          261 |    19608 | {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200} |   394.927 |
| patient    | A_replication | A_DNN        |     0.670185 |       0.00361098 |    0.224336 |       0.669849 |              0.657395 |               0.68147  |      0.219389 | 0.226968 |  0.625255 |   0.176195 |  0.625224 | 0.274916 |   0.625259 |     0.550334 |  0.715983 |   0.196549 |  0.485637 | 0.27984  |   0.745512 |    12957 |     4423 |     1146 |     1082 |          0.708866 |          261 |    19608 | {"dropout": 0.4, "hidden": [128, 64]}                       |   603.382 |

## 7. Methodological limitations
- The paper does not publish complete feature specifications or exact hyperparameter search grids.
- The adult cohort is approximated by excluding age bins [0-10) and [10-20), which also excludes some individuals aged 18–19.
- The paper's reported test size of 20,354 equals 20% of the original 101,766 encounters, creating ambiguity about whether its reported results used the stated cohort exclusions.
- The DNN architecture and training details are not fully specified in the paper.
- Full-cohort sensitivity analysis was not performed.
- Numerical similarity or difference in ROC-AUC does not establish exact methodological replication.


## 8. Audit of inconsistencies in the published paper

Several numerical and methodological inconsistencies in the published
paper should be considered when interpreting the replication.

### 8.1 XGBoost positive-class F1-score

The paper's discussion reports an XGBoost positive-class F1-score of
0.89, whereas its results table reports 0.032. These values are
substantially different and cannot both represent the same metric,
model, and evaluation setting.

### 8.2 XGBoost confusion matrix and precision

The paper reports 2,285 positive test cases, 2,244 false negatives,
and 1,172 false positives in its XGBoost confusion matrix.

These values imply:

- True positives = 2,285 - 2,244 = 41
- Precision = 41 / (41 + 1,172) = approximately 0.0338

This does not agree with the positive-class precision of 0.437
reported in the paper's results table.

### 8.3 Test-set size and cohort exclusions

The paper reports 20,354 test encounters, which is exactly 20% of
the original 101,766 encounters.

Our restricted cohort contains 98,493 encounters and produces
19,699 test encounters under an 80/20 split.

This creates ambiguity about whether the paper's reported test
results were calculated using the original dataset or the
post-exclusion cohort. The available reporting does not resolve
this ambiguity.

### 8.4 Interpretation

These discrepancies are reported as internal numerical and
methodological inconsistencies in the published article.
They are not evidence of intentional wrongdoing.

They limit the ability to identify a single unambiguous set of
paper results against which an exact replication can be verified.
Accordingly, this project compares against the paper's reported
values while explicitly documenting these limitations.

## 9. Interpretation
The encounter-level and patient-level experiments were completed using four models. Patient-level splitting prevents the same patient from appearing in both training and test sets. Results from the two protocols should be interpreted separately because their evaluation designs differ.

The results constitute a documented replication-oriented implementation, not an exact reproduction of every paper detail.

## 10. Final status
Primary encounter and patient experiments completed. Full-cohort sensitivity analysis not performed. No additional model improvements included.