from data.paired_image_dataset import Paired_Image_Dataset
from torchvision.transforms import Compose

from iNAS.data.transform import Identity, RandomHorizontalFlip, Resize, ToTensor
from iNAS.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Saliency_Dataset(Paired_Image_Dataset):
    """[summary]

    Args:
        opt ([type]): [description]
    """

    def __init__(self, opt):
        super(Saliency_Dataset, self).__init__(opt)
        self.opt = opt
        if self.opt['phase'] == 'train':
            self.transform = Compose([
                Resize(base_size=(opt['input_size'], opt['input_size'])),
                RandomHorizontalFlip() if opt['use_flip'] else Identity(),
                ToTensor(mean=opt['mean'], std=opt['std'])
            ])
        else:
            self.transform = Compose([
                Resize(base_size=(opt['input_size'], opt['input_size']), image_only=True),
                ToTensor(mean=opt['mean'], std=opt['std'], image_only=True)
            ])
