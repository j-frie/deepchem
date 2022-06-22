import unittest
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

  @pytest.mark.torch
  def test_validation_torch(self):
    model = dc.models.MultitaskClassifier(n_tasks=2,
                                          n_features=1024,
                                          dropouts=0.5)
    self.validation_callback(model)

  @pytest.mark.tensorflow
  def test_validation_keras(self):
    model = dc.models.RobustMultitaskClassifier(n_tasks=2,
                                                n_features=1024,
                                                dropouts=0.5)
    self.validation_callback(model)
