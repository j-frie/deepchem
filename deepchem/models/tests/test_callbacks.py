import unittest
from abc import ABC

import pytest
import tempfile
import deepchem as dc
import numpy as np
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO


class TestCallbacks(unittest.TestCase):

  def validation_callback(self, model):
    """Test ValidationCallback for given model.

    The model can either be a TorchModel or a KerasModel.
    It has to be a classifier with n_tasks = 2 and n_features = 1024."""

    tasks, datasets, transformers = dc.molnet.load_clintox()
    train_dataset, valid_dataset, test_dataset = datasets

    # Train the model while logging the validation ROC AUC.

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
    log = StringIO()
    save_dir = tempfile.mkdtemp()
    callback = dc.models.ValidationCallback(valid_dataset,
                                            30, [metric],
                                            log,
                                            save_dir=save_dir,
                                            save_on_minimum=False,
                                            transformers=transformers)
    model.fit(train_dataset, callbacks=callback)

    # Parse the log to pull out the AUC scores.
    log.seek(0)
    scores = []
    for line in log:
      score = float(line.split('=')[-1])
      scores.append(score)

    # The last reported score should match the current performance of the model.
    valid_score = model.evaluate(valid_dataset, [metric], transformers)
    self.assertAlmostEqual(valid_score['mean-roc_auc_score'],
                           scores[-1],
                           places=5)

    # The highest recorded score should match get_best_score().

    self.assertAlmostEqual(max(scores), callback.get_best_score(), places=5)

    # Reload the save model and confirm that it matches the best logged score.

    model.restore(model_dir=save_dir)
    valid_score = model.evaluate(valid_dataset, [metric], transformers)
    self.assertAlmostEqual(valid_score['mean-roc_auc_score'],
                           max(scores),
                           places=5)

    # Make sure get_best_score() still works when save_dir is not specified

    callback = dc.models.ValidationCallback(valid_dataset,
                                            30, [metric],
                                            log,
                                            save_on_minimum=False,
                                            transformers=transformers)
    model.fit(train_dataset, callbacks=callback)
    log.seek(0)
    scores = []
    for line in log:
      score = float(line.split('=')[-1])
      scores.append(score)

    self.assertTrue(abs(max(scores) - callback.get_best_score()) < 0.05)

  def vc_with_regression_loss(self, model):
    """Helper function to test loss calculation during validation callback.

    The loss is calculated in the validation callback using the train_dataset
    and compared to the loss returned by model.fit()."""
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP',
                                                           splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    metrics = [dc.metrics.Metric(dc.metrics.rms_score)]
    log = StringIO()
    vc_train = dc.models.ValidationCallback(train_dataset,
                                            interval=1,
                                            metrics=metrics,
                                            output_file=log,
                                            transformers=[],
                                            prefix_logging='train',
                                            loss_logging=True,
                                            model_task_type='regression')

    # loss returned by model.fit is calculated before updating weights
    # loss in callback is calculated after updating weights
    # --> the callback loss has to compared to the loss at the next step
    loss_before_step = model.fit(train_dataset,
                                 nb_epoch=1,
                                 callbacks=[vc_train])
    loss_after_step = model.fit(train_dataset, nb_epoch=1)
    # Parse the log to pull out the loss
    log.seek(0)
    scores = []
    for line in log:
      score = float(line.split('=')[-1])
      scores.append(score)

    self.assertEqual(len(scores), 1)
    self.assertAlmostEqual(scores[0], loss_after_step, delta=0.01)

  @pytest.mark.torch
  def test_validation_torch(self):
    model = dc.models.MultitaskClassifier(n_tasks=2,
                                          n_features=1024,
                                          dropouts=0.5)
    self.validation_callback(model)

  @pytest.mark.torch
  def test_validation_loss_regression_torch(self):
    model = dc.models.MultitaskRegressor(n_tasks=1,
                                         n_features=1024,
                                         layer_sizes=[1024, 1],
                                         dropouts=0.0,
                                         batch_normalize=False,
                                         batch_size=902)
    self.vc_with_regression_loss(model)

  @pytest.mark.tensorflow
  def test_validation_keras(self):
    model = dc.models.RobustMultitaskClassifier(n_tasks=2,
                                                n_features=1024,
                                                dropouts=0.5)
    self.validation_callback(model)

  @pytest.mark.tensorflow
  def test_validation_loss_regression_keras(self):
    import tensorflow as tf

    class ToyKerasModel(dc.models.KerasModel):

      def __init__(self, n_features=1024, **kwargs):
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_features, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        super().__init__(keras_model,
                         loss=dc.models.losses.L2Loss(),
                         output_types=['prediction'],
                         **kwargs)

    model = ToyKerasModel(n_features=1024, batch_size=902)
    self.vc_with_regression_loss(model)
