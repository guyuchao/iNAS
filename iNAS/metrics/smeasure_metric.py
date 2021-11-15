import numpy as np
import torch

from iNAS.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class Smeasure:
    """[summary]
    """

    def __init__(self):
        self.alpha = 0.5
        self.q = []

    def S_object(self, pred, gt):

        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)

        o_fg = self.object(fg, gt)
        o_bg = self.object(bg, 1 - gt)

        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def S_region(self, pred, gt):
        X, Y = self.centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self.divideGT(gt, X, Y)
        p1, p2, p3, p4 = self.dividePrediction(pred, X, Y)
        Q1 = self.ssim(p1, gt1)
        Q2 = self.ssim(p2, gt2)
        Q3 = self.ssim(p3, gt3)
        Q4 = self.ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            Q = alpha / (beta + 1e-20)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def mask_normalize(self, mask):
        # input 'mask': HxW
        # output: HxW [0,255]
        return mask / (np.max(mask) + 1e-8)

    def add(self, pred, gt):
        pred = self.mask_normalize(pred)
        gt = self.mask_normalize(gt)

        if (len(pred.shape) > 2):
            pred = pred[:, :, 0]
        if (len(gt.shape) > 2):
            gt = gt[:, :, 0]

        gt = torch.from_numpy(gt).view(1, 1, *gt.shape).float()
        pred = torch.from_numpy(pred).view(1, 1, *pred.shape).float()

        y = gt.mean()
        if y <= 1e-5:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            Q = self.alpha * self.S_object(pred, gt) + (1 - self.alpha) * self.S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        if torch.isnan(Q):
            print('nan')
        else:
            self.q.append(Q.item())

    def add_batch(self, preds, gts):
        for pred, gt in zip(preds, gts):
            self.add(pred, gt)

    def get_metric(self):
        avg_q = sum(self.q) / len(self.q)
        return avg_q
