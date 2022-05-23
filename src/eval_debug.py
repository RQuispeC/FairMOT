from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils.sysconfig import get_config_h_filename

import _init_paths
import os
import os.path as osp
import cv2
import json
import logging
import argparse
import motmetrics as mm
import numpy as np
import random
import torch

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from datasets.dataset_factory import get_dataset

from tracking_utils.utils import mkdir_if_missing
from opts import opts
from tracking_utils.log import logger

def main():
    torch.manual_seed(137)
    torch.manual_seed(137)
    torch.cuda.manual_seed(137)
    np.random.seed(137)
    random.seed(137)

    # run tracking
    accs = []

    seq_name = 'test_seq'
    
    data_root = '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/fairmot-results/noted-javelin/mot20dev_train_test_split6/test'
    gt_filename = data_root + '/MOT20-01-img1_gt.txt'
    result_filename = data_root + '/MOT20-01-img1_pred.txt'
    '''
    data_root = '/mnt/c/Users/edquispe/OneDrive - Microsoft/Documents/mot/fairmot-results/cunning-flea/split5-train-test/test'
    gt_filename = data_root + '/images-64pm_cafe_shop_0_camera_1_gt.txt'
    result_filename = data_root + '/images-64pm_cafe_shop_0_camera_1_pred.txt'
    '''
    data_type = 'mot'
    seqs = [seq_name]


    # eval
    logger.info('Evaluate seq: {}'.format(seq_name))
    evaluator = Evaluator(data_root, seq_name, data_type, gt_filename=gt_filename)
    print(len(evaluator.gt_frame_dict))
    print(len(evaluator.gt_ignore_frame_dict))
    print(evaluator.gt_frame_dict[1])
    print("ignored:", sum([len(v) for v in evaluator.gt_ignore_frame_dict.values()]))
    accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)


if __name__ == '__main__':
    main()

