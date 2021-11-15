import numpy as np
import time
import torch

from iNAS.utils.ptflops import get_model_complexity_info


def get_net_Gflops_Mparams(network, spatial_size):
    """[summary]

    Args:
        network ([type]): [description]
        spatial_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(spatial_size, tuple):
        spatial_size = (spatial_size, spatial_size)
    flops, params = get_model_complexity_info(network, (3, *spatial_size), as_strings=False, print_per_layer_stat=False)
    flops /= 1e9
    params /= 1e6
    return {'gflops': flops, 'mparams': params}


class EvalSpeed():
    """[summary]

    Args:
        device ([type]): [description]
        img_size ([type]): [description]
    """

    def __init__(self, device, img_size):
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.device = device
        if device == 'cuda':
            bs = 32
        else:
            bs = 1
        self.inp = torch.randn(bs, 3, *img_size)

    def eval(self, model):
        if self.device == 'cuda':
            model = model.cuda()
            model.eval()
            inputs = self.inp.cuda()
            return self.computeTimeCuda(model, inputs)
        else:
            model.eval()
            inputs = self.inp
            return self.computeTimeCpu(model, inputs)

    @torch.no_grad()
    def computeTimeCuda(self, model, inputs):
        bs = inputs.shape[0]
        time_spent = []

        for _ in range(30):
            _ = model(inputs)
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

        for _ in range(70):
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            time_spent.append(time.time() - start_time)
        return np.mean(time_spent) * 1000 / bs

    @torch.no_grad()
    def computeTimeCpu(self, model, inputs):
        bs = inputs.shape[0]
        time_spent = []

        for _ in range(30):
            _ = model(inputs)

        for _ in range(70):
            start_time = time.time()
            _ = model(inputs)
            time_spent.append(time.time() - start_time)
        return np.mean(time_spent) * 1000 / bs
