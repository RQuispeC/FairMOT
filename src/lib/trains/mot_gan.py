from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer_gan import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))


    def forward(self, output, batch):
        # normal MOT loss
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        if not self.opt.mse_loss:
            output['hm'] = _sigmoid(output['hm'])

        hm_loss += self.crit(output['hm'], batch['hm'])
        if self.opt.wh_weight > 0:
            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'],
                batch['ind'], batch['wh'])

        if self.opt.reg_offset and self.opt.off_weight > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                        batch['ind'], batch['reg'])

        if self.opt.id_weight > 0:
            id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
            id_head = id_head[batch['reg_mask'] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['ids'][batch['reg_mask'] > 0]

            id_output = self.classifier(id_head).contiguous()
            if self.opt.id_loss == 'focal':
                id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1, id_target.long().view(-1, 1), 1)
                id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,alpha=0.25, gamma=2.0, reduction="sum") / id_output.size(0)
            else:
                id_loss += self.IDLoss(id_output, id_target)

        det_loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss + self.opt.off_weight * off_loss
        if self.opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        # adversarial loss
        is_real_label_reid = torch.ones_like(output['d_r_fake_reid_logits']).to(output['d_r_fake_reid_logits'].device)
        is_real_label_detect = torch.ones_like(output['d_d_fake_detect_logits']).to(output['d_d_fake_detect_logits'].device)
        loss_adv_reid = F.binary_cross_entropy_with_logits(output['d_r_fake_reid_logits'], is_real_label_reid)
        loss_adv_detect = F.binary_cross_entropy_with_logits(output['d_d_fake_detect_logits'], is_real_label_detect)
        loss += self.opt.adv_weight * (loss_adv_reid + loss_adv_detect)

        loss_stats = {'loss_generator': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss, 'loss_gen_reid': loss_adv_reid, 'loss_gen_detect': loss_adv_detect}
        return loss, loss_stats

class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, opt, idname, task_specific_logit_name, task_generic_logit_name):
        super(DiscriminatorLoss, self).__init__()
        self.idname = idname
        self.task_specific_logit_name = task_specific_logit_name
        self.task_generic_logit_name = task_generic_logit_name

    def forward(self, output):
        is_real_label = torch.ones_like(output[self.task_specific_logit_name]).to(output[self.task_specific_logit_name].device)
        is_fake_label = torch.zeros_like(output[self.task_generic_logit_name]).to(output[self.task_generic_logit_name].device)

        loss_disc_specific = F.binary_cross_entropy_with_logits(output[self.task_specific_logit_name], is_real_label)
        loss_disc_generic = F.binary_cross_entropy_with_logits(output[self.task_generic_logit_name], is_fake_label)

        loss = loss_disc_specific + loss_disc_generic
        loss_stats = {self.idname: loss}
        return loss, loss_stats

class MotTrainerGan(BaseTrainer):
    def __init__(self, opt, model, logger):
        super(MotTrainerGan, self).__init__(opt, model, logger)
        optimizer_generator = torch.optim.Adam(model.generator.parameters(), opt.lr)
        optimizer_reid_discr = torch.optim.Adam(model.discr_reid.parameters(), opt.lr)
        optimizer_detect_discr = torch.optim.Adam(model.discr_detect.parameters(), opt.lr)
        self.optimizers = {}
        self.optimizers['generator'] = optimizer_generator
        self.optimizers['reid_discr'] = optimizer_reid_discr
        self.optimizers['detect_discr'] = optimizer_detect_discr
        self.loss_generator = MotLoss(opt)
        self.loss_discriminator_reid = DiscriminatorLoss(opt, 'disc_reid', 'd_r_real_reid_logits', 'd_r_real_shared_logits')
        self.loss_discriminator_detect = DiscriminatorLoss(opt, 'disc_detect', 'd_d_real_detect_logits', 'd_d_real_shared_logits')
        self.optimizers['generator'].add_param_group({'params': self.loss_generator.parameters()})
        self.optimizers['reid_discr'].add_param_group({'params': self.loss_discriminator_reid.parameters()})
        self.optimizers['detect_discr'].add_param_group({'params': self.loss_discriminator_detect.parameters()})
        self.loss_states = ['loss_generator', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'disc_reid', 'disc_detect']
        self.step = 0

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model = DataParallel(
                self.model, device_ids=gpus, 
                chunk_sizes=chunk_sizes).to(device)
            self.loss_generator = DataParallel(
                self.loss_generator, device_ids=gpus, 
                chunk_sizes=chunk_sizes).to(device)
            self.loss_discriminator_reid = DataParallel(
                self.loss_discriminator_reid, device_ids=gpus, 
                chunk_sizes=chunk_sizes).to(device)
            self.loss_discriminator_detect = DataParallel(
                self.loss_discriminator_detect, device_ids=gpus, 
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model = self.model.to(device)
            self.loss_generator = self.loss_generator.to(device)
            self.loss_discriminator_reid = self.loss_discriminator_reid.to(device)
            self.loss_discriminator_detect = self.loss_discriminator_detect.to(device)
        
        for k in self.optimizers.keys():
            for state in self.optimizers[k].state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=device, non_blocking=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

    def run_epoch(self, phase, epoch, data_loader):
        if phase == 'train':
            self.model.train()
            self.loss_generator.train()
        else:
            if len(self.opt.gpus) > 1:
                self.model.module.eval()
            else:
                self.model.eval()
            torch.cuda.empty_cache()

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_states}
        num_iters = len(data_loader) if self.opt.num_iters < 0 else self.opt.num_iters
        bar = Bar('{}/{}'.format(self.opt.task, self.opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            self.step += 1
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)

            if phase == 'train':
                # generator
                output = self.model(batch['input'], mode='train_generator')
                loss, loss_stats = self.loss_generator(output, batch)
                loss = loss.mean()
                self.optimizers['generator'].zero_grad()
                loss.backward()
                self.optimizers['generator'].step()

                # discriminators
                output = self.model(batch['input'], mode='train_discriminator')

                loss_dict_reid, loss_stats_reid = self.loss_discriminator_reid(output)
                loss_dict_reid = loss_dict_reid.mean()
                loss_stats.update(loss_stats_reid)
                
                loss_dict_detect, loss_stats_detect = self.loss_discriminator_detect(output)
                loss_dict_detect = loss_dict_detect.mean()
                loss_stats.update(loss_stats_detect)
                
                self.optimizers['reid_discr'].zero_grad()
                self.optimizers['detect_discr'].zero_grad()
                loss_dict_reid.backward(retain_graph=True)
                loss_dict_detect.backward()
                self.optimizers['reid_discr'].step()
                self.optimizers['detect_discr'].step()
            else:
                output = self.model(batch, mode='eval')

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            if phase == 'train':
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                    self.logger.scalar_summary('train_inner/Loss_{}'.format(l), loss_stats[l].mean().item(), self.step)
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not self.opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if self.opt.print_iter > 0:
                if iter_id % self.opt.print_iter == 0:
                    print('{}/{}| {}'.format(self.opt.task, self.opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            if self.opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch
            
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results