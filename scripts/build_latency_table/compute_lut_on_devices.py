import argparse
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict

from iNAS.archs.iNAS_standalone.iNAS_backbone import MBConvLayer
from iNAS.archs.iNAS_standalone.iNAS_util import (BatchNorm2d, Conv2d, DecoderFusion, DecoderUpsample, Identity,
                                                  TransportDownsample, TransportFusion, TransportUpsample)


class Latency_Measure:
    """[summary]

    Args:
        latency_table ([type]): [description]
        device ([type]): [description]
        save_root ([type]): [description]
    """

    def __init__(self, latency_table, device, save_root):
        self.latency_table = latency_table
        self.device = device
        self.save_root = save_root
        self.ind = 0

    def get_input(self, shape, type='tensor'):
        if self.device == 'cuda':
            bs = 32
        else:
            bs = 1

        if type == 'tensor':
            t = torch.randn(bs, *shape)
            if self.device == 'cuda':
                return t.cuda()
            else:
                return t
        else:
            raise NotImplementedError

    def compute_latency_cpu_gpu(self, model, inputs):
        if self.device == 'cpu':
            return self.compute_latency_cpu(model, inputs)
        else:
            return self.compute_latency_cuda(model, inputs)

    def compute_latency_cpu(self, model, inputs):
        bs = inputs.shape[0]

        model.eval()

        time_spent = []

        for idx in range(25):
            with torch.no_grad():
                _ = model(inputs)

        for idx in range(25):
            start_time = time.time()
            with torch.no_grad():
                _ = model(inputs)
            time_spent.append(time.time() - start_time)

        time_spent = [t * 1000 / bs for t in time_spent]

        return time_spent

    def compute_latency_cuda(self, model, inputs):
        bs = inputs.shape[0]

        if self.device == 'cuda':
            model = model.cuda()

        model.eval()

        time_spent = []

        for idx in range(50):
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize()  # warmup

        for idx in range(50):
            start_time = time.time()
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
            time_spent.append(time.time() - start_time)

        time_spent = [t * 1000 / bs for t in time_spent]
        torch.cuda.empty_cache()  #
        time.sleep(
            5)  # cuda test speed slower when not empty cache for previous run, sleep because asynchronous need time
        return time_spent

    def parse_strcfg2dictcfg(self, strcfg):
        dictcfg = {}
        int_list = ['inres', 'outres', 'inc', 'outc', 'stride', 'kernel', 'fusion', 'ratio']
        for item in strcfg.strip().split():
            k, v = item.split(':')[0], item.split(':')[1]
            if k in int_list:
                dictcfg[k] = int(v)
            else:
                dictcfg[k] = v
        return dictcfg

    def update_stem_latency(self, dictcfg):
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        inp = self.get_input(inp_shape)
        module = nn.Sequential(
            OrderedDict([('stemconv', Conv2d(3, dictcfg['outc'], kernel_size=3, stride=2, padding=1)),
                         ('stemnorm', BatchNorm2d(dictcfg['outc'])), ('stemact', nn.SiLU(inplace=True))]))

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_dynamicmb_latency(self, dictcfg):
        module = MBConvLayer(
            in_channels=dictcfg['inc'],
            out_channels=dictcfg['outc'],
            stride=dictcfg['stride'],
            expand_ratio=dictcfg['ratio'],
            kernel_size_array=[dictcfg['kernel']])
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_fusiontp_latency(self, dictcfg):
        inp_shape = (dictcfg['fusion'], dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        module = TransportFusion(dictcfg['inc'], kernel_size_array=[dictcfg['kernel']])
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_fusiond_latency(self, dictcfg):
        inp_shape = (dictcfg['fusion'], dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        module = DecoderFusion(dictcfg['inc'], kernel_size_array=[dictcfg['kernel']])
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_branchtp_latency(self, dictcfg):
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        if dictcfg['inres'] == dictcfg['outres']:
            module = Identity()
        elif dictcfg['inres'] < dictcfg['outres']:
            level_diff = int(math.log2(dictcfg['outres'] / dictcfg['inres']))
            module = TransportUpsample(level_diff, in_channels=dictcfg['inc'], out_channels=dictcfg['outc'])
        else:
            level_diff = int(math.log2(dictcfg['inres'] / dictcfg['outres']))
            module = TransportDownsample(level_diff, in_channels=dictcfg['inc'], out_channels=dictcfg['outc'])
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_branchd_latency(self, dictcfg):
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        if dictcfg['inres'] == dictcfg['outres']:
            module = Identity()
        elif dictcfg['inres'] < dictcfg['outres']:
            level_diff = int(math.log2(dictcfg['outres'] / dictcfg['inres']))
            module = DecoderUpsample(level_diff, in_channels=dictcfg['inc'], out_channels=dictcfg['outc'])
        else:
            raise NotImplementedError
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_decoder_latency(self, dictcfg):
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        module = nn.Sequential(
            OrderedDict([
                ('conv', Conv2d(dictcfg['inc'], 1, kernel_size=1)),
                ('act', nn.Sigmoid()),
                ('upsample', nn.UpsamplingBilinear2d(scale_factor=2)),
            ]))
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def update_upsample_latency(self, dictcfg):
        inp_shape = (dictcfg['inc'], dictcfg['inres'], dictcfg['inres'])
        module = nn.UpsamplingBilinear2d(scale_factor=2)
        inp = self.get_input(inp_shape)

        if self.device == 'mobile':
            module.eval()
            script_model = torch.jit.trace(module, inp)
            save_path = os.path.join(self.save_root, 'mobile_jitmodels/%d_%s.pt' % (self.ind, str((1, *inp_shape))))
            save_path = save_path.replace(' ', '').replace('(', '').replace(')', '')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            script_model.save(save_path)
            self.ind += 1
        else:
            latency_list = self.compute_latency_cpu_gpu(module, inp)
            dictcfg['latency_mean'] = np.mean(latency_list)
            dictcfg['latency_var'] = np.var(latency_list)
            return dictcfg

    def run_latency(self):
        self.result_list = []
        for idx, latency_cfg in enumerate(self.latency_table):
            dictcfg = self.parse_strcfg2dictcfg(latency_cfg)
            if dictcfg['name'] == 'stem':
                resdict = self.update_stem_latency(dictcfg)
            elif dictcfg['name'] == 'dynamicmb':
                resdict = self.update_dynamicmb_latency(dictcfg)
            elif dictcfg['name'] == 'branchtp':
                resdict = self.update_branchtp_latency(dictcfg)
            elif dictcfg['name'] == 'fusiontp':
                resdict = self.update_fusiontp_latency(dictcfg)
            elif dictcfg['name'] == 'branchd':
                resdict = self.update_branchd_latency(dictcfg)
            elif dictcfg['name'] == 'fusiond':
                resdict = self.update_fusiond_latency(dictcfg)
            elif dictcfg['name'] == 'decoder':
                resdict = self.update_decoder_latency(dictcfg)
            else:
                print(dictcfg['name'])
                raise NotImplementedError

            print('finish[%d/%d]' % (idx, len(self.latency_table)))

            if resdict is not None:
                self.result_list.append(
                    latency_cfg.format(latency_mean=resdict['latency_mean'], latency_var=resdict['latency_var']))
                print(self.result_list[-1])

        if len(self.result_list):
            with open(os.path.join(self.save_root, 'latency_lookup_table_%s.txt' % self.device), 'w') as fw:
                fw.writelines(self.result_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mobile'], default='cpu')
    parser.add_argument(
        '--lut_template_path',
        type=str,
        default='experiments/pretrained_models/resource_tables/latency/lut_template.txt',
        help='Save root path.')
    parser.add_argument(
        '--save_root',
        type=str,
        default='experiments/pretrained_models/resource_tables/latency/',
        help='Save root path.')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    with open(args.lut_template_path) as fr:
        latency_table = fr.readlines()
    Latency_Measure(latency_table, args.device, args.save_root).run_latency()
