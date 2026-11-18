"""Research-grade pipeline (replication + improved experiments).

Modules
-------
config    : paths, seeds, constants
data      : loading, verification, cohort definition, label, splits
features  : replication feature set (FS-R) and engineered feature set (FS-E)
metrics   : metric computation (threshold-dependent + paper-style weighted/class-wise)
keras_mlp : scikit-learn compatible Keras DNN classifier
models    : model factories + hyper-parameter grids
experiment: generic tune -> OOF threshold -> refit -> test evaluation -> persist
"""
