from tensorflow.python.training import session_run_hook
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer, _as_graph_element
import numpy as np


class LoggingTensorHook(session_run_hook.SessionRunHook):
  """Prints the given tensors once every N local steps or once every N seconds.
  The tensors will be printed to the log, with `INFO` severity.
  """

  def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
               formatter=None):
    """Initializes a LoggingHook monitor.
    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
          seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
          provided.
      formatter: function, takes dict of `tag`->`Tensor` and returns a string.
          If `None` uses default printing all tensors.
    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_iter and every_n_secs must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError("invalid every_n_iter=%s." % every_n_iter)
    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = tensors.keys()
    self._tensors = tensors
    self._formatter = formatter
    self._timer = SecondOrStepTimer(every_secs=every_n_secs,
                                    every_steps=every_n_iter)

  def begin(self):
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      return SessionRunArgs(self._current_tensors)
    else:
      return None

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      original = np.get_printoptions()
      np.set_printoptions(suppress=True)
      elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
      if self._formatter:
        logging.info(self._formatter(run_values.results))
      else:
        stats = []
        for tag in self._tag_order:
          stats.append("%s = %s" % (tag, run_values.results[tag]))
        if elapsed_secs is not None:
          logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
        else:
          logging.info("%s", ", ".join(stats))
      np.set_printoptions(**original)
    self._iter_count += 1
    