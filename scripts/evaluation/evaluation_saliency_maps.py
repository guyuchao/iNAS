import argparse
import cv2
import os
import torch
from glob import glob
from tqdm import tqdm

from iNAS.metrics.fmeasure_metric import Fmeasure
from iNAS.metrics.mae_metric import MAE
from iNAS.metrics.smeasure_metric import Smeasure

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='datasets/fastsaliency/ECSSD')
    parser.add_argument('--pred_dir', type=str, default='results/minarch/')
    parser.add_argument('--metric', type=str, default='Fmeasure')
    args = parser.parse_args()

    metric_dict = {'Fmeasure': Fmeasure, 'MAE': MAE, 'Smeasure': Smeasure}

    metric_evaluator = metric_dict[args.metric]()
    gt_list = sorted(glob(os.path.join(args.image_dir, '*.png')))
    pred_list = sorted(glob(os.path.join(args.pred_dir, '*.png')))

    for pred_path, gt_path in tqdm(zip(pred_list, gt_list)):
        pred_name = os.path.basename(pred_path).split('.')[0]
        gt_name = os.path.basename(gt_path).split('.')[0]
        assert pred_name == gt_name
        pred = cv2.imread(pred_path)
        gt = cv2.imread(gt_path)
        metric_evaluator.add(pred, gt)
    print('%s: %.8f' % (args.metric, metric_evaluator.get_metric()))
