from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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




def write_results(out_filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(out_filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save predictions to {}'.format(out_filename))

def write_gt_formated(out_filename, img_file_paths):
    def load_raw_label(label_path: str, img_size: tuple) -> np.array:
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
        h, w = img_size
        labels = labels0.copy()
        labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
        labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
        labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
        labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        return labels
    
    def format_labels(frame_id, bbox_xyxy, identities):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        out_str = ""
        for xyxy, track_id in zip(bbox_xyxy, identities):
            if track_id < 0 or track_id is None:
                continue
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h)
            out_str += line
        return out_str

    with open(out_filename, 'w') as fout:
        for i, f_path in enumerate(img_file_paths):
            label_path = f_path.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
            cur_img = cv2.imread(f_path)
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            targets = load_raw_label(label_path, cur_img.shape[:2]) if os.path.exists(label_path) else None

            fout.write(format_labels(frame_id=(i + 1),
                            bbox_xyxy=targets[:, 2:],
                            identities=targets[:, 1]))
    logger.info('saved ground truth to {}'.format(out_filename))

def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, result_filename, save_dir):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=opt.frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps) | {}'.format(frame_id, 1. / max(1e-5, timer.average_time), path))

        # run tracking
        timer.tic()
        if opt.gpus[0] >= 0:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if opt.save_images or opt.show_image:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if opt.show_image:
            cv2.imshow('online_im', online_im)
        if opt.save_images:
            cv2.imwrite(osp.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results)
    #write_results_score(result_filename, results)
    return frame_id, timer.average_time, timer.calls

def get_sequences(root_data_path, eval_data_file_name):
    seq_data = {}
    with open(eval_data_file_name, 'r') as file:
        img_files = file.readlines()
        img_files = [osp.join(root_data_path, x.strip()) for x in img_files]
        for img_file in img_files:
            seq_name = '-'.join(img_file.split('/')[-3:-1])
            if seq_name not in seq_data:
                seq_data[seq_name] = []
            seq_data[seq_name].append(img_file)
    return seq_data

def attach_eval_dataset_params(opt):
    class Struct:
        def __init__(self, entries):
            for k, v in entries.items():
                self.__setattr__(k, v)
    default_dataset_info = {'default_resolution': [608, 1088], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278], 'nID': 14455}
    dataset = Struct(default_dataset_info)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    return opt

def main(opt):
    torch.manual_seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    data_root = opt.data_root_dir
    # assume that test set is only of 1 dataset
    dataset_name = None
    dataset_file_paths = None
    for k, v in json.load(open(opt.data_cfg))['test'].items():
        dataset_name = k
        dataset_file_paths = v
    print('Setting up data for validation for', dataset_name)
    seqs = get_sequences(opt.data_root_dir, dataset_file_paths)
    opt = attach_eval_dataset_params(opt)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq_name, seq_data in seqs.items():
        output_dir = osp.join(opt.save_dir, seq_name)
        if not osp.isdir(output_dir):
            os.makedirs(output_dir)

        # get predictions for video
        logger.info('Prediction seq: {}'.format(seq_name))
        dataloader = datasets.LoadImages(seq_data, opt.img_size)
        result_filename = osp.join(opt.save_dir, '{}_pred.txt'.format(seq_name))
        nf, ta, tc = eval_seq(opt, dataloader, result_filename, save_dir=output_dir)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # load ground truths and format them similar to predictions
        gt_filename = osp.join(opt.save_dir, '{}_gt.txt'.format(seq_name))
        write_gt_formated(gt_filename, seq_data)

        # eval
        logger.info('Evaluate seq: {}'.format(seq_name))
        evaluator = Evaluator(data_root, seq_name, data_type, gt_filename=gt_filename)
        accs.append(evaluator.eval_file(result_filename))
        if opt.save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq_name))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

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
    Evaluator.save_summary(summary, osp.join(opt.save_dir, 'summary.xlsx'))


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)

