import torch.nn as nn

from iNAS.archs.iNAS_supernet.iNAS_backbone import DynamicBackbone
from iNAS.archs.iNAS_supernet.iNAS_decoder import DynamicDecoder
from iNAS.archs.iNAS_supernet.iNAS_transport import DynamicTransport
from iNAS.archs.iNAS_supernet.uniform_sampler import Uniform_Architecture_Sampler
from iNAS.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class iNASSupernet(nn.Module):
    """[summary]

    Args:
        cfg ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, cfg=None, num_classes=1, activation='none', output_level_ids=(0, 1, 2, 3, 4)):
        super(iNASSupernet, self).__init__()
        if cfg is None:
            self.cfg = Uniform_Architecture_Sampler().sampling('max')
        else:
            self.cfg = cfg

        self.backbone = DynamicBackbone(self.cfg['backbone_cfg'])

        level_channels_list = {
            'level_0': self.cfg['backbone_cfg']['stage_1']['filter'][-1],
            'level_1': self.cfg['backbone_cfg']['stage_2']['filter'][-1],
            'level_2': self.cfg['backbone_cfg']['stage_3']['filter'][-1],
            'level_3': self.cfg['backbone_cfg']['stage_5']['filter'][-1],
            'level_4': self.cfg['backbone_cfg']['stage_7']['filter'][-1],
        }

        self.transport = DynamicTransport(self.cfg['transport_cfg'], level_channels_list)
        self.decoder = DynamicDecoder(self.cfg['decoder_cfg'], level_channels_list, num_classes, activation,
                                      output_level_ids)

    def forward(self, batched_inputs):
        if isinstance(batched_inputs, dict):
            images = batched_inputs['image']
            if 'cfg' in batched_inputs:
                cfg = batched_inputs['cfg']
            else:
                cfg = self.cfg
        else:
            images = batched_inputs
            cfg = self.cfg
        features = self.backbone(images, cfg['backbone_cfg'])
        features = self.transport(features, cfg['transport_cfg'])
        results = self.decoder(features, cfg['decoder_cfg'])

        return results
