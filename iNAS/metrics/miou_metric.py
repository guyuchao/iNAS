import torch

from iNAS.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class MIOU(object):

    def __init__(self, ignore_label=255, num_classes=19):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).cuda()

    def add_batch(self, preds, gts):
        device = preds.device
        gts = gts.to(device=device)
        preds = torch.argmax(preds, dim=1)
        keep = gts != self.ignore_label
        self.hist += torch.bincount(
            gts[keep] * self.num_classes + preds[keep],
            minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def get_metric(self):
        ious = self.hist.diag() / (self.hist.sum(dim=0) + self.hist.sum(dim=1) - self.hist.diag())
        miou = ious.mean()
        return miou
