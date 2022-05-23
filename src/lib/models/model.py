from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_dla_dcn_split_1 import get_pose_net as get_dla_dcn_split_1
from .networks.pose_dla_dcn_split_2 import get_pose_net as get_dla_dcn_split_2
from .networks.pose_dla_dcn_split_3 import get_pose_net as get_dla_dcn_split_3
from .networks.pose_dla_dcn_split_4 import get_pose_net as get_dla_dcn_split_4
from .networks.pose_dla_dcn_split_5 import get_pose_net as get_dla_dcn_split_5
from .networks.pose_dla_dcn_split_6 import get_pose_net as get_dla_dcn_split_6
from .networks.pose_dla_dcn_split_6_gan import get_pose_net as get_dla_dcn_split_6_gan
from .networks.pose_dla_dcn_split_7 import get_pose_net as get_dla_dcn_split_7
from .networks.pose_dla_dcn_split_8 import get_pose_net as get_dla_dcn_split_8
from .networks.pose_dla_dcn_split_9 import get_pose_net as get_dla_dcn_split_9
from .networks.pose_dla_dcn_2 import get_pose_net as get_dla_dcn_2
from .networks.pose_dla_dcn_2_gan import get_pose_net as get_dla_dcn_2_gan
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_fpn_dcn import get_pose_net as get_pose_net_fpn_dcn
from .networks.pose_hrnet import get_pose_net as get_pose_net_hrnet
from .networks.pose_dla_conv import get_pose_net as get_dla_conv
from .yolo import get_pose_net as get_pose_net_yolo

_model_factory = {
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dla2': get_dla_dcn_2,
  'dlasplit1': get_dla_dcn_split_1,
  'dlasplit2': get_dla_dcn_split_2,
  'dlasplit3': get_dla_dcn_split_3,
  'dlasplit4': get_dla_dcn_split_4,
  'dlasplit5': get_dla_dcn_split_5,
  'dlasplit6': get_dla_dcn_split_6,
  'dlasplit6gan': get_dla_dcn_split_6_gan,
  'dlasplit7': get_dla_dcn_split_7,
  'dlasplit8': get_dla_dcn_split_8,
  'dlasplit9': get_dla_dcn_split_9,
  'dla2gan': get_dla_dcn_2_gan,
  'dlaconv': get_dla_conv,
  'resdcn': get_pose_net_dcn,
  'resfpndcn': get_pose_net_fpn_dcn,
  'hrnet': get_pose_net_hrnet,
  'yolo': get_pose_net_yolo
}

def create_model(arch, heads, head_conv):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

# save and load for gan assume multiple optimizers

def load_gan_model(model, model_path, optimizers, resume=False, 
               lr=None, lr_step=None, load_on_generator=False):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  if load_on_generator:
    model_state_dict = model.generator.state_dict()
  else:
    model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  if load_on_generator:
    model.generator.load_state_dict(state_dict, strict=False)
  else:
    model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizers is not None and resume:
    if 'optimizers' in checkpoint:
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      loaded_optimizers = True
      for optimizer in optimizers.keys():
        if optimizer in checkpoint['optimizers']:
          optimizers[optimizer].load_state_dict(checkpoint['optimizers'][optimizer])
          for param_group in optimizers[optimizer].param_groups:
            param_group['lr'] = start_lr
          print('Resumed optimizer `{}` with start lr `{}`'.format(optimizer, start_lr))
        else:
          print('optimizer `{}` is not avaialable in checkpoint'.format(optimizer))
    else:
      print('No optimizers parameters in checkpoint.')
  if optimizers is not None:
    return model, optimizers, start_epoch
  else:
    return model

def save_gan_model(path, epoch, model, optimizers):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  data['optimizers'] = {}
  for optimizer_key, optimizer_value in optimizers.items():
    data['optimizers'][optimizer_key] = optimizer_value.state_dict()
  torch.save(data, path)
