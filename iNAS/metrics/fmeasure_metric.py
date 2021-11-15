import numpy as np

from iNAS.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class Fmeasure:
    """[summary]
    """

    def __init__(self):
        self.beta = 0.3
        self.pre = []
        self.rec = []

    def compute_pre_rec(self, gt, mask, mybins=np.arange(0, 256)):
        assert len(gt.shape) >= 2 and len(mask.shape) >= 2, 'gt or mask is not matrix!'
        assert gt.shape == mask.shape, 'The shapes of gt and mask are different!'

        if (len(gt.shape) > 2):  # convert to one channel
            gt = gt[:, :, 0]
        if (len(mask.shape) > 2):  # convert to one channel
            mask = mask[:, :, 0]

        gtNum = gt[gt > 128].size  # pixel number of ground truth foreground regions
        pp = mask[gt > 128]  # mask predicted pixel values in the ground truth foreground region
        nn = mask[gt <= 128]  # mask predicted pixel values in the ground truth background region

        pp_hist, pp_edges = np.histogram(
            pp, bins=mybins
        )  # count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
        nn_hist, nn_edges = np.histogram(nn, bins=mybins)

        pp_hist_flip = np.flipud(
            pp_hist
        )  # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
        nn_hist_flip = np.flipud(nn_hist)

        pp_hist_flip_cum = np.cumsum(
            pp_hist_flip
        )  # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
        nn_hist_flip_cum = np.cumsum(nn_hist_flip)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-8)  # TP/(TP+FP)
        recall = pp_hist_flip_cum / (gtNum + 1e-8)  # TP/(TP+FN)

        precision[np.isnan(precision)] = 0.0
        recall[np.isnan(recall)] = 0.0

        return np.reshape(precision, (len(precision))), np.reshape(recall, (len(recall)))

    def mask_normalize(self, mask):
        # input 'mask': HxW
        # output: HxW [0,255]
        return mask / (np.max(mask) + 1e-8) * 255.0

    def add_batch(self, preds, gts):
        for pred, gt in zip(preds, gts):
            self.add(pred, gt)

    def add(self, pred, gt):
        pred = self.mask_normalize(pred)
        gt = self.mask_normalize(gt)
        pre, rec = self.compute_pre_rec(gt, pred, mybins=np.arange(0, 256))
        self.pre.append(pre)
        self.rec.append(rec)

    def get_metric(self):
        nums = len(self.pre)
        pre = sum(self.pre) / (nums + 1e-8)
        rec = sum(self.rec) / (nums + 1e-8)
        fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec + 1e-8)
        return max(fm)
