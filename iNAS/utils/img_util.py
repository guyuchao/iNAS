import cv2
import os
import torch
import torch.nn.functional as F


def imwrite(img, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, img)


def img2tensor(imgs, bgr2rgb=True, float32=True):

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, height, width, min_max=(0, 1)):
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = output * 255
    output = output.type(torch.uint8).cpu().numpy()
    output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR)
    return output


def tensor2img_batch(tensor, height_list, width_list, min_max=(0, 1)):
    """[summary]
    Args:
        tensor ([type]): [description]
        height_list ([type]): [description]
        width_list ([type]): [description]
        min_max (tuple, optional): [description]. Defaults to (0, 1).
    Returns:
        [type]: [description]
    """
    outputs = tensor.detach().clamp_(*min_max).permute(0, 2, 3, 1)
    outputs = outputs * 255
    outputs = outputs.type(torch.uint8).cpu().numpy()
    output_list = []
    for output, height, width in zip(outputs, height_list, width_list):
        output_list.append(cv2.resize(output, (width, height), interpolation=cv2.INTER_LINEAR))
    return output_list


def batch_resize(tensor, height_list, width_list):
    """[summary]
    Args:
        tensor ([type]): [description]
        height_list ([type]): [description]
        width_list ([type]): [description]
        min_max (tuple, optional): [description]. Defaults to (0, 1).
    Returns:
        [type]: [description]
    """
    tensor = tensor.detach()
    anchor_height = height_list[0]
    assert all([h == anchor_height for h in height_list]), 'need same height to resize'

    anchor_width = width_list[0]
    assert all([w == anchor_width for w in width_list]), 'need same width to resize'

    tensor = F.interpolate(tensor, size=(anchor_height, anchor_width), mode='bilinear', align_corners=True)
    return tensor
