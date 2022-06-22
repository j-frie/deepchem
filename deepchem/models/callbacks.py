"""
Callback functions that can be invoked while fitting a KerasModel or TorchModel.
"""
import sys

tf_installed = False
torch_installed = False
try:
  from deepchem.models.keras_model import KerasModel
  tf_installed = True
except ModuleNotFoundError:
  pass
try:
  from deepchem.models.torch_models import TorchModel
  import torch
  torch_installed = True
except ModuleNotFoundError:
  pass


class ValidationCallback(object):
  """Performs validation while training a KerasModel or TorchModel.

  This is a callback that can be passed to fit().  It periodically computes a
  set of metrics over a validation set, writes them to a file, and keeps track
  of the best score. In addition, it can save the best model parameters found
  so far to a directory on disk, updating them every time it finds a new best
  validation score.

  If Tensorboard logging is enabled on the KerasModel, the metrics are also
  logged to Tensorboard.  This only happens when validation coincides with a
  step on which the model writes to the log.  You should therefore make sure
  that this callback's reporting interval is an even fraction or multiple of
  the model's logging interval.
  """

  def __init__(self,
               dataset,
               interval,
               metrics,
               output_file=sys.stdout,
               save_dir=None,
               save_metric=0,
               save_on_minimum=True,
               transformers=[],
               prefix_logging='eval',
               loss_logging=False):
    """Create a ValidationCallback.

    Parameters
    ----------
    dataset: dc.data.Dataset
      the validation set on which to compute the metrics
    interval: int
      the interval (in training steps) at which to perform validation
    metrics: list of dc.metrics.Metric
      metrics to compute on the validation set
    output_file: file
      to file to which results should be written
    save_dir: str
      if not None, the model parameters that produce the best validation score
      will be written to this directory
    save_metric: int
      the index of the metric to use when deciding whether to write a new set
      of parameters to disk
    save_on_minimum: bool
      if True, the best model is considered to be the one that minimizes the
      validation metric.  If False, the best model is considered to be the one
      that maximizes it.
    transformers: List[Transformer]
      List of `dc.trans.Transformer` objects. These transformations
      must have been applied to `dataset` previously. The dataset will
      be untransformed for metric evaluation.
    prefix_logging: str
      prefix used for wandb logging, default: 'eval'
    loss_logging: bool
      if True, the loss of the current model on the given dataset is calculated
    """
    self.dataset = dataset
    self.interval = interval
    self.metrics = metrics
    self.output_file = output_file
    self.save_dir = save_dir
    self.save_metric = save_metric
    self.save_on_minimum = save_on_minimum
    self._best_score = None
    self.transformers = transformers
    self.prefix_logging = prefix_logging
    self.loss_logging = loss_logging

  def __call__(self, model, step):
    """This is invoked by the KerasModel/TorchModel after every step of fitting.

    Parameters
    ----------
    model: KerasModel or TorchModel
      the model that is being trained
    step: int
      the index of the training step that has just completed
    """
    if step % self.interval != 0:
      return
    scores = model.evaluate(self.dataset, self.metrics, self.transformers)
    message = 'Step %d validation:' % step
    for key in scores:
      message += ' %s=%g' % (key, scores[key])
    print(message, file=self.output_file)
    if model.tensorboard:
      for key in scores:
        model._log_scalar_to_tensorboard(key, scores[key],
                                         model.get_global_step())
    score = scores[self.metrics[self.save_metric].name]
    if not self.save_on_minimum:
      score = -score
    if self._best_score is None or score < self._best_score:
      self._best_score = score
      if self.save_dir is not None:
        model.save_checkpoint(model_dir=self.save_dir)
    # train_loss is calculated during training and logged as 'loss'
    if self.loss_logging:
      scores['vc_loss'] = self._calculate_loss(model)
    if model.wandb_logger is not None:
      # Log data to WandB
      data = {f'{self.prefix_logging}/{k}': v for k, v in scores.items()}
      model.wandb_logger.log_data(data, step)

  def get_best_score(self):
    """This getter returns the best score evaluated on the given dataset.

    Returns
    -------
    float
      The best score.
    """
    if self.save_on_minimum:
      return self._best_score
    else:
      return -self._best_score

  def _calculate_loss(self, model):
    """Use model to predict output and calculate current loss on given dataset.

    Returns
    float
      The current loss of the given dataset.
    """
    if torch_installed and isinstance(model, TorchModel):
      outputs = torch.from_numpy(model.predict(self.dataset))
      labels = torch.from_numpy(self.dataset.y)
      weights = torch.from_numpy(self.dataset.w)
      return model._loss_fn([outputs], [labels], [weights]).item()

    elif tf_installed and isinstance(model, KerasModel):
      outputs = model.predict(self.dataset)
      labels = self.dataset.y
      weights = self.dataset.w
      return float(model._loss_fn([outputs], [labels], [weights]))
