import numpy as np

from iNAS.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class MAE:
    """[summary]
    """

    def __init__(self):
        self.mae = []

    def mask_normalize(self, mask):
        # input 'mask': HxW
        # output: HxW [0,255]
        return mask / (np.max(mask) + 1e-8)

    def add(self, pred, gt):
        h, w = pred.shape

        pred = self.mask_normalize(pred)
        gt = self.mask_normalize(gt)

        sumError = np.sum(np.absolute((pred.astype(float) - gt.astype(float))))
        maeError = sumError / (float(h) * float(w) + 1e-8)

        self.mae.append(maeError)

    def get_metric(self):
        mae = sum(self.mae) / len(self.mae)
        return mae

    def add_batch(self, preds, gts):
        for pred, gt in zip(preds, gts):
            self.add(pred, gt)
