import argparse
import cv2
import glob
import os
import os.path as osp
import torch
from tqdm import tqdm

from iNAS.archs.iNAS_standalone_arch import iNASStandalone
from iNAS.utils.img_util import img2tensor, imwrite, tensor2img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='datasets/fastsaliency/ECSSD')
    parser.add_argument('--model_config', type=str, default='CPU_search/CPU_lat@27.00ms_Fmeasure@0.9437.json')
    parser.add_argument('--model_path', type=str, default='CPU_search/CPU_lat@27.00ms_Fmeasure@0.9437.pth')
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--mean', type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=list, default=[0.229, 0.224, 0.225])
    args = parser.parse_args()

    model_name = osp.splitext(osp.basename(args.model_config))[0]
    result_root = f'results/{model_name}'
    os.makedirs(result_root, exist_ok=True)

    # set up the RIDNet
    net = iNASStandalone(args.model_config, deploy=True).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint)
    net.eval()

    # scan all the jpg and png images
    img_list = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))

    for idx, img_path in enumerate(tqdm(img_list)):
        img_name = os.path.basename(img_path).split('.')[0]
        # read image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        img = cv2.resize(img, (args.image_size, args.image_size))
        img = img2tensor(img, bgr2rgb=True, float32=True) / 255

        mean = torch.as_tensor(args.mean, dtype=img.dtype, device=img.device)[:, None, None]
        std = torch.as_tensor(args.std, dtype=img.dtype, device=img.device)[:, None, None]
        img.sub_(mean).div_(std)
        img = img.unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            output = net({'image': img})
        # save image
        output = tensor2img(output, height, width, min_max=(0, 1))
        save_img_path = os.path.join(result_root, f'{img_name}.png')
        imwrite(output, save_img_path)
