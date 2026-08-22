"""
src/models/trainer.py — Model Training Pipeline
===================================================

WHY THIS FILE EXISTS:
    Training a neural network requires configuring an optimizer,
    loss function, metrics, class weights, and callbacks.
    This class handles the entire training orchestration.

HOW IT WORKS:
    Runs a hand-written training loop (GradientTape) over batches cut
    straight from the memmapped .npy arrays instead of `model.fit()`.

    WHY NOT model.fit(): during Day 4, `fit()` hung indefinitely at 0% CPU
    on this platform, through three different input pipelines
    (`keras.utils.Sequence`, `PyDataset`, and
    `tf.data.Dataset.from_generator(...).prefetch(...)`). The suspicion at
    the time was that Keras's background prefetch thread deadlocked against
    the memmap reads.

    That diagnosis turned out to be WRONG. The real cause was an abseil
    symbol collision between TensorFlow and Apache Arrow — see the note in
    `src/models/__init__.py`. Every one of those hangs happened because
    `src.models.__init__` imported the sklearn-dependent `evaluator` before
    the TF-dependent `lstm_model`, so TF was poisoned before `fit()` was
    ever reached. The input pipeline was never the problem.

    The manual loop is kept anyway, deliberately:
      - it is written, tested, and demonstrably works (~36 ms/batch);
      - it makes class weighting, early stopping, LR reduction, and
        checkpointing explicit and inspectable rather than hidden in
        framework callbacks;
      - it removes an entire class of background-thread behaviour from the
        training path.
    `fit()` has NOT been re-benchmarked since the real fix landed. If a
    future change wants it back, prove it completes first.

    Because Keras callbacks only run inside `fit()`, early stopping, LR
    reduction, and best-weight checkpointing are implemented inline below —
    same semantics, ~20 lines of explicit state.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam

from src.models.lstm_model import PredictiveMaintenanceModel
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# How often (in batches) to log intra-epoch progress. The manual loop has no
# Keras progress bar, so without this a 2700-batch epoch looks like a hang.
LOG_EVERY_N_BATCHES = 100


def iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (X_batch, y_batch) as float32 arrays, covering every sample once.

    X and y are typically memmaps of several GB, so each batch is
    materialised only when it is yielded. Shuffled indices are sorted
    *within* a batch: the epoch order is still random, but the disk reads
    backing one batch stay monotonic, which matters a lot for memmaps.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        if shuffle:
            batch_idx = np.sort(batch_idx)

        yield (
            np.asarray(X[batch_idx], dtype=np.float32),
            np.asarray(y[batch_idx], dtype=np.float32).reshape(-1),
        )


class ModelTrainer:
    """Handles compilation and training of the predictive model."""

    def __init__(self, model: PredictiveMaintenanceModel):
        self.model_wrapper = model
        self.model = model.model
        self.optimizer: Optional[Adam] = None
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.history: Optional[Dict[str, List[float]]] = None

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset."""
        n_samples = len(y)
        n_pos = np.sum(np.asarray(y) == 1)
        n_neg = n_samples - n_pos

        if n_pos == 0:
            logger.warning("No positive samples found! Using equal weights.")
            return {0: 1.0, 1: 1.0}

        # Inverse ratio weighting
        weight_0 = (1 / n_neg) * (n_samples / 2.0)
        weight_1 = (1 / n_pos) * (n_samples / 2.0)

        weights = {0: float(weight_0), 1: float(weight_1)}
        logger.info(
            f"Computed class weights: Class 0: {weight_0:.2f}, Class 1: {weight_1:.2f}"
        )
        return weights

    def compile(self, learning_rate: float = 0.001) -> None:
        """Set up the optimizer and loss used by the manual training loop."""
        try:
            self.optimizer = Adam(learning_rate=learning_rate)
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()

            # Keras still needs to be compiled for save()/load() to round-trip
            # the training config, even though fit() is never called.
            self.model.compile(
                optimizer=self.optimizer,
                loss="binary_crossentropy",
                metrics=[
                    AUC(name="auc"),
                    Precision(name="precision"),
                    Recall(name="recall"),
                ],
            )
            logger.info("Model compiled successfully with Adam and BinaryCrossentropy.")
        except Exception as e:
            logger.error(f"Failed to compile model: {str(e)}")
            raise ModelTrainingError(f"Model compilation failed: {str(e)}")

    def _make_train_step(self):
        """
        Build the graph-compiled train step.

        Built once per train() call so the tf.function traces against the
        current optimizer; the None batch dimension keeps the final short
        batch of each epoch from triggering a retrace.
        """
        seq_len = self.model.input_shape[1]
        n_features = self.model.input_shape[2]

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, seq_len, n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ]
        )
        def train_step(x_batch, y_batch, sample_weights):
            with tf.GradientTape() as tape:
                preds = self.model(x_batch, training=True)
                loss = self.loss_fn(y_batch, preds, sample_weight=sample_weights)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss, preds

        return train_step

    def _run_validation(
        self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int
    ) -> Dict[str, float]:
        """Forward-pass the validation set and return loss + imbalance metrics."""
        metrics = {
            "auc": AUC(name="val_auc"),
            "precision": Precision(name="val_precision"),
            "recall": Recall(name="val_recall"),
        }
        loss_sum, n_seen = 0.0, 0

        for x_batch, y_batch in iter_batches(X_val, y_val, batch_size=batch_size):
            preds = self.model(x_batch, training=False)
            loss = self.loss_fn(y_batch, preds)

            loss_sum += float(loss) * len(x_batch)
            n_seen += len(x_batch)
            for metric in metrics.values():
                metric.update_state(y_batch, tf.reshape(preds, [-1]))

        results = {name: float(metric.result()) for name, metric in metrics.items()}
        results["loss"] = loss_sum / max(n_seen, 1)
        return results

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        checkpoint_path: str = "models/best_model.keras",
        patience: int = 5,
        lr_patience: int = 3,
        min_lr: float = 1e-6,
    ) -> Dict[str, List[float]]:
        """
        Execute the training loop.

        Returns a history dict (``{"loss": [...], "auc": [...], ...}``) rather
        than a Keras ``History`` object, since fit() is not used.
        """
        if self.optimizer is None:
            raise ModelTrainingError("compile() must be called before train().")

        try:
            logger.info(
                f"Starting training for {epochs} epochs (Batch size: {batch_size})"
            )

            class_weights = self._compute_class_weights(y_train)
            w_neg, w_pos = class_weights[0], class_weights[1]

            has_val = X_val is not None and y_val is not None
            train_step = self._make_train_step()

            history: Dict[str, List[float]] = {
                "loss": [],
                "auc": [],
                "precision": [],
                "recall": [],
            }
            if has_val:
                history.update(
                    {
                        "val_loss": [],
                        "val_auc": [],
                        "val_precision": [],
                        "val_recall": [],
                    }
                )

            # Early stopping / checkpointing track val AUC when we have a
            # validation set (higher is better), and training loss otherwise.
            best_score = -np.inf if has_val else np.inf
            best_weights = None
            epochs_without_improvement = 0
            lr_wait = 0
            best_lr_loss = np.inf

            train_metrics = {
                "auc": AUC(name="auc"),
                "precision": Precision(name="precision"),
                "recall": Recall(name="recall"),
            }

            for epoch in range(1, epochs + 1):
                for metric in train_metrics.values():
                    metric.reset_state()

                loss_sum, n_seen = 0.0, 0

                for batch_num, (x_batch, y_batch) in enumerate(
                    iter_batches(
                        X_train, y_train, batch_size, shuffle=True, seed=epoch
                    ),
                    start=1,
                ):
                    sample_weights = np.where(y_batch == 1, w_pos, w_neg).astype(
                        np.float32
                    )
                    loss, preds = train_step(x_batch, y_batch, sample_weights)

                    loss_sum += float(loss) * len(x_batch)
                    n_seen += len(x_batch)
                    for metric in train_metrics.values():
                        metric.update_state(y_batch, tf.reshape(preds, [-1]))

                    if batch_num % LOG_EVERY_N_BATCHES == 0:
                        logger.info(
                            f"  Epoch {epoch} | batch {batch_num} | "
                            f"loss {loss_sum / n_seen:.4f} | "
                            f"auc {float(train_metrics['auc'].result()):.4f}"
                        )

                epoch_loss = loss_sum / max(n_seen, 1)
                history["loss"].append(epoch_loss)
                for name, metric in train_metrics.items():
                    history[name].append(float(metric.result()))

                msg = (
                    f"Epoch {epoch}/{epochs} — loss: {epoch_loss:.4f} | "
                    f"auc: {history['auc'][-1]:.4f} | "
                    f"precision: {history['precision'][-1]:.4f} | "
                    f"recall: {history['recall'][-1]:.4f}"
                )

                if has_val:
                    val = self._run_validation(X_val, y_val, batch_size)
                    for name, value in val.items():
                        history[f"val_{name}"].append(value)
                    msg += (
                        f" || val_loss: {val['loss']:.4f} | "
                        f"val_auc: {val['auc']:.4f} | "
                        f"val_precision: {val['precision']:.4f} | "
                        f"val_recall: {val['recall']:.4f}"
                    )

                logger.info(msg)

                # --- Checkpoint + early stopping -----------------------------
                current = history["val_auc"][-1] if has_val else epoch_loss
                improved = current > best_score if has_val else current < best_score

                if improved:
                    best_score = current
                    best_weights = [w.copy() for w in self.model.get_weights()]
                    epochs_without_improvement = 0
                    self._save_checkpoint(checkpoint_path, epoch, best_score, has_val)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(no improvement for {patience} epochs)."
                        )
                        break

                # --- ReduceLROnPlateau on the loss we are minimising ----------
                plateau_loss = history["val_loss"][-1] if has_val else epoch_loss
                if plateau_loss < best_lr_loss:
                    best_lr_loss = plateau_loss
                    lr_wait = 0
                else:
                    lr_wait += 1
                    if lr_wait >= lr_patience:
                        old_lr = float(self.optimizer.learning_rate.numpy())
                        new_lr = max(old_lr * 0.5, min_lr)
                        if new_lr < old_lr:
                            self.optimizer.learning_rate.assign(new_lr)
                            logger.info(
                                f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}"
                            )
                        lr_wait = 0

            if best_weights is not None:
                self.model.set_weights(best_weights)
                logger.info("Restored best weights.")

            self.history = history
            logger.info("Training completed.")
            return history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ModelTrainingError(f"Training failed: {str(e)}")

    def _save_checkpoint(
        self, checkpoint_path: str, epoch: int, score: float, has_val: bool
    ) -> None:
        """Persist the current (best-so-far) weights."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        monitored = "val_auc" if has_val else "loss"
        logger.info(
            f"Epoch {epoch}: {monitored} improved to {score:.4f} — saved {path}"
        )
