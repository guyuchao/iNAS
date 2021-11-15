import torch
import torch.nn as nn
from collections import OrderedDict

from iNAS.archs.iNAS_standalone.iNAS_backbone import Backbone
from iNAS.archs.iNAS_standalone.iNAS_decoder import Decoder
from iNAS.archs.iNAS_standalone.iNAS_transport import Transport
from iNAS.archs.iNAS_supernet.uniform_sampler import Uniform_Architecture_Sampler
from iNAS.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class iNASStandalone(nn.Module):
    """[summary]

    Args:
        cfg ([type]): [description]
        deploy (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, cfg, deploy=False, num_classes=1, activation='sigmoid', output_level_ids='(0, 1, 2, 3, 4)'):
        super(iNASStandalone, self).__init__()
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            self.cfg = Uniform_Architecture_Sampler().load_arch(cfg)
        self.backbone = Backbone(self.cfg['backbone_cfg'])
        level_channels_list = {
            'level_0': self.cfg['backbone_cfg']['stage_1']['filter'][-1],
            'level_1': self.cfg['backbone_cfg']['stage_2']['filter'][-1],
            'level_2': self.cfg['backbone_cfg']['stage_3']['filter'][-1],
            'level_3': self.cfg['backbone_cfg']['stage_5']['filter'][-1],
            'level_4': self.cfg['backbone_cfg']['stage_7']['filter'][-1],
        }

        self.transport = Transport(self.cfg['transport_cfg'], level_channels_list)
        self.decoder = Decoder(self.cfg['decoder_cfg'], level_channels_list, num_classes, activation, output_level_ids)

        if deploy:
            self.reparams()

    @torch.no_grad()
    def reparams(self):
        for _, module in self.backbone.named_modules():
            if hasattr(module, 'reparams'):
                module.reparams()
        for _, module in self.transport.named_modules():
            if hasattr(module, 'reparams'):
                module.reparams()
        for _, module in self.decoder.named_modules():
            if hasattr(module, 'reparams'):
                module.reparams()

    def get_substart_end(self, kernel_size, max_kernel_size):
        start = (max_kernel_size - kernel_size) // 2
        end = start + kernel_size
        return start, end

    def load_supernet_weight(self, state_dict):
        this_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        for k in this_state_dict.keys():

            this_shape = this_state_dict[k].shape

            if len(this_shape) == 0:
                pass
            elif len(this_shape) == 1:
                new_state_dict[k] = state_dict[k][:this_shape[0]]
            elif len(this_shape) == 2:
                new_state_dict[k] = state_dict[k][:this_shape[0], :this_shape[1]]
            elif len(this_shape) == 4:
                start, end = self.get_substart_end(this_shape[-1], state_dict[k].shape[-1])
                new_state_dict[k] = state_dict[k][:this_shape[0], :this_shape[1], start:end, start:end]
            else:
                pass
        self.load_state_dict(new_state_dict, strict=True)
        self.reparams()

    def forward(self, batched_inputs):
        if isinstance(batched_inputs, dict):
            images = batched_inputs['image']
        else:
            images = batched_inputs
        features = self.backbone(images)
        features = self.transport(features)

        results = self.decoder(features)
        return results
