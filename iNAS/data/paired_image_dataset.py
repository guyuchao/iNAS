import cv2
import os.path as osp
from torch.utils.data import Dataset


class Paired_Image_Dataset(Dataset):

    def __init__(self, opt):
        super(Paired_Image_Dataset, self).__init__()

        self.data_dir = opt['data_dir']
        data_list_path = opt['data_list']
        self.paths = list()

        with open(data_list_path, 'r') as lines:
            for line in lines:
                line_arr = line.split()
                image_path = osp.join(self.data_dir, line_arr[0].strip())
                label_path = osp.join(self.data_dir, line_arr[1].strip())
                self.paths.append((image_path, label_path))

        self.transform = None
        self.label_map = None

    def __getitem__(self, idx):
        image_path, label_path = self.paths[idx]

        # assume label is gray scale image
        image, label = cv2.imread(image_path)[:, :, ::-1], cv2.imread(label_path, 0)

        image_name = image_path.split('/')[-1]
        height, width = image.shape[:2]

        if self.label_map is not None:
            label = self.label_map[label]

        data_dict = dict(
            image=image,
            label=label,
            name=image_name,
            height=height,
            width=width,
            image_path=image_path,
            label_path=label_path)

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.paths)
