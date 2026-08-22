"""
ML Models Package
=================
Provides the LSTM model architecture, training pipeline, and evaluator.

IMPORT ORDER IS LOAD-BEARING — DO NOT ALPHABETISE THESE IMPORTS.

    TensorFlow must be imported before scikit-learn / pandas.

WHY: TensorFlow statically links its own copy of abseil, and so does Apache
Arrow (`libarrow.*.dylib`), which pandas and scikit-learn both load. Whichever
library loads first wins the `AbslInternalPerThreadSemWait` symbol for the whole
process. If Arrow wins, TensorFlow's `absl::Mutex::Block()` ends up waiting on
Arrow's per-thread semaphore, which never signals it — so the first TF graph
execution deadlocks at 0% CPU, with no traceback, no timeout, and no error.

Confirmed by sampling a hung process:

    tensorflow::...::RunSync
      absl::Notification::WaitForNotification()      (libtensorflow_framework)
        absl::Mutex::Block()                         (libtensorflow_framework)
          AbslInternalPerThreadSemWait_lts_20250814  (libarrow.2400.dylib)
            ^-- the wrong abseil

`evaluator` imports sklearn.metrics; `lstm_model` imports tensorflow. Importing
evaluator first is exactly the poisoned order, and it is what hung
`scripts/train_model.py`. See docs/Day4.md.

Any entry point that uses TensorFlow must therefore import `src.models` (or
tensorflow itself) before importing `src.data`, which pulls in pandas.
`tests/conftest.py` does this for the test suite.
"""

# isort: skip_file  (isort would alphabetise these and reintroduce the deadlock)

# TensorFlow first — see the note above.
from src.models.lstm_model import PredictiveMaintenanceModel
from src.models.trainer import ModelTrainer

# Only now is it safe to pull in scikit-learn (which loads libarrow).
from src.models.evaluator import ModelEvaluator

__all__ = ["PredictiveMaintenanceModel", "ModelTrainer", "ModelEvaluator"]
