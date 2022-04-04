from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .mot_gan import MotTrainerGan


train_factory = {
  'mot': MotTrainer,
  'mot_gan': MotTrainerGan
}
