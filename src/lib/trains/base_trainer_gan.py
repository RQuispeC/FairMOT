from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch, mode='train_generator'):
    outputs = self.model(batch['input'])
    if 'train' not in mode:
      return outputs[-1]
    else:
      loss, loss_stats = self.loss(outputs, batch, mode)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, logger):
    self.opt = opt
    self.model = model
    self.logger = logger
    self.optimizers = None

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model = DataParallel(
        self.model, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model = self.model.to(device)
    
    for k in self.optimizers.keys():
      for state in self.optimizers[k].state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    raise NotImplementedError
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)