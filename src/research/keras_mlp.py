"""scikit-learn compatible Keras deep neural network (DNN / MLP) classifier.

Makes the DNN usable inside sklearn Pipelines and GridSearchCV so that it is
tuned with exactly the same 5-fold CV protocol as the other model families.
Early stopping uses an internal validation split carved from the data passed
to `fit` (i.e. from the training fold only), never from the test set.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class KerasDNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(128, 64), dropout=0.3, l2=0.0, batch_norm=False,
                 learning_rate=1e-3, batch_size=512, epochs=60, patience=6,
                 class_weight="balanced", val_fraction=0.1, random_state=42, verbose=0):
        self.hidden = hidden
        self.dropout = dropout
        self.l2 = l2
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def _build(self, n_features: int):
        import tensorflow as tf
        reg = tf.keras.regularizers.l2(self.l2) if self.l2 else None
        layers = [tf.keras.layers.Input(shape=(n_features,))]
        for units in self.hidden:
            layers.append(tf.keras.layers.Dense(units, kernel_regularizer=reg))
            if self.batch_norm:
                layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.Activation("relu"))
            if self.dropout:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        model = tf.keras.Sequential(layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
        return model

    def fit(self, X, y, sample_weight=None):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                              random_state=self.random_state)
        cw = None
        if self.class_weight == "balanced":
            pos = ytr.mean()
            cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
        self.model_ = self._build(X.shape[1])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=self.patience,
                                              restore_best_weights=True)
        hist = self.model_.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=self.epochs,
                               batch_size=self.batch_size, class_weight=cw, callbacks=[es],
                               verbose=self.verbose)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.n_epochs_ = len(self.history_["loss"])
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype=np.float32), verbose=0, batch_size=4096).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ---- pickling support (joblib) -------------------------------------- #
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                state["model_"].save(path)
                with open(path, "rb") as fh:
                    state["model_"] = fh.read()
        return state

    def __setstate__(self, state):
        if isinstance(state.get("model_"), (bytes, bytearray)):
            import tensorflow as tf
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.keras")
                with open(path, "wb") as fh:
                    fh.write(state["model_"])
                state["model_"] = tf.keras.models.load_model(path)
        self.__dict__.update(state)
