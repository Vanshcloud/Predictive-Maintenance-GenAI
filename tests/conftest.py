"""
tests/conftest.py — pytest session bootstrap
=============================================

WHY THIS FILE EXISTS:
    TensorFlow must be imported before pandas or scikit-learn, because those
    two load Apache Arrow (`libarrow.*.dylib`) and Arrow statically links its
    own copy of abseil — as does TensorFlow. Whichever library loads first
    wins the `AbslInternalPerThreadSemWait` symbol for the entire process.

    If Arrow wins, TensorFlow's `absl::Mutex::Block()` waits on Arrow's
    per-thread semaphore, which never signals it. The first `tf.function`
    execution then deadlocks: 0% CPU, forever, with no traceback, no timeout,
    and no error message. It is indistinguishable from "slow".

    Confirmed by sampling a hung process:

        tensorflow::...::RunSync
          absl::Notification::WaitForNotification()      (libtensorflow_framework)
            absl::Mutex::Block()                         (libtensorflow_framework)
              AbslInternalPerThreadSemWait_lts_20250814  (libarrow.2400.dylib)
                ^-- the wrong abseil

    Reproduced deterministically outside pytest:

        pandas .rolling() -> import tensorflow -> train step   # hangs
        import tensorflow -> pandas .rolling() -> train step   # fine

    conftest.py is imported before any test module, so importing TF here fixes
    the order for the whole session. Without it, running
    tests/unit/test_preprocessing.py (pandas/sklearn) before
    tests/unit/test_model.py hangs the suite indefinitely.

    Do not remove this import, and do not "clean up" the noqa.
    See also: src/models/__init__.py and docs/Day4.md.
"""

# Imported purely for its side effect: it fixes abseil symbol resolution order.
import tensorflow as tf  # noqa: F401
