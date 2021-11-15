from copy import deepcopy

from iNAS.utils import get_root_logger
from iNAS.utils.registry import RESOURCE_REGISTRY


def build_resource(opt):
    """Build resource calculator from options.

    Args:
        opt (dict): Configuration.
    """
    opt = deepcopy(opt)
    resource_type = opt.pop('type')
    model = RESOURCE_REGISTRY.get(resource_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Resource calculator [{model.__class__.__name__}] is created.')
    return model


@RESOURCE_REGISTRY.register()
class LatencyCalculator():
    """[summary]

    Args:
        resource_table ([type]): [description]
        device ([type]): [description]
    """

    def __init__(self, resource_table, device):
        self.dynamicmb_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} ' \
            'outc:{outc} stride:{stride} kernel:{kernel} ratio:{ratio}'
        self.fusion_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} ' \
            'outc:{outc} fusion:{fusion} kernel:{kernel}'
        self.branch_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} outc:{outc}'

        with open(resource_table, 'r') as fr:
            latency_list = fr.readlines()
            self.latency_dict = self.build_latency_dict(latency_list)
        self.device = device

    def build_latency_dict(self, latency_list):
        latency_dict = {}
        for row in latency_list:
            name = row.split('latency_mean')[0].strip()
            for item in row.split(' '):
                kn, vn = item.split(':')[0], item.split(':')[1]
                if kn == 'latency_mean':
                    latency_dict[name] = float(vn)
        return latency_dict

    def cal_stage_latency(self, stage_name, stage_cfg, inc, inp_res):
        if 'stem' in stage_name:
            arch_name = 'stem'
        elif 'stage' in stage_name:
            arch_name = 'dynamicmb'
        else:
            raise NotImplementedError
        latency = 0.0
        depth = stage_cfg['depth']
        kernel_array = stage_cfg['kernel']
        filter_array = stage_cfg['filter']
        stride = stage_cfg['stride']
        ratio = stage_cfg['ratio']

        for d in range(depth):
            if stride == 2:
                out_res = inp_res // 2
            else:
                out_res = inp_res
            outc = filter_array[d]
            kernel = max(kernel_array[d])
            latency += self.latency_dict[self.dynamicmb_template.format(
                name=arch_name,
                inres=inp_res,
                outres=out_res,
                inc=inc,
                outc=outc,
                ratio=ratio,
                stride=stride,
                kernel=kernel)]
            inc = outc
            inp_res = out_res
            stride = 1
        return latency

    def cal_backbone_latency(self, inp_res, arch_cfg):
        latency = 0.0
        inc = 3

        for stage_name, stage_cfg in arch_cfg.items():
            latency += self.cal_stage_latency(stage_name, stage_cfg, inc, inp_res)
            if stage_cfg['stride'] == 2:
                inp_res = inp_res // 2

            inc = stage_cfg['filter'][-1]

        return latency

    def cal_latency_transport(self, inp_res, arch_cfg, level_channels_dict):

        base_res = inp_res // 2
        latency = 0.0

        for level_name, level_cfg in arch_cfg.items():
            conn_list = level_cfg['connection']
            kernel = max(level_cfg['kernel'])
            level_idx = int(level_name.split('_')[1])
            outres = base_res // (2**level_idx)

            outc = level_channels_dict['level_%d' % level_idx]
            for conn_idx in conn_list:
                name = 'branchtp'
                inres = base_res // (2**conn_idx)
                inc = level_channels_dict['level_%d' % conn_idx]
                latency += self.latency_dict[self.branch_template.format(
                    name=name, inres=inres, outres=outres, inc=inc, outc=outc)]
            name = 'fusiontp'
            latency += self.latency_dict[self.fusion_template.format(
                name=name, inres=outres, outres=outres, inc=outc, outc=outc, fusion=len(conn_list), kernel=kernel)]
        return latency

    def cal_latency_decoder(self, base_res, arch_cfg, level_channels_dict):

        base_res = base_res // 2
        latency = 0.0

        for level_name, level_cfg in arch_cfg.items():
            conn_list = level_cfg['connection']
            kernel = max(level_cfg['kernel'])
            level_idx = int(level_name.split('_')[1])
            outres = base_res // (2**level_idx)

            outc = level_channels_dict['level_%d' % level_idx]
            for conn_idx in conn_list:
                name = 'branchd'
                inres = base_res // (2**conn_idx)
                inc = level_channels_dict['level_%d' % conn_idx]
                latency += self.latency_dict[self.branch_template.format(
                    name=name, inres=inres, outres=outres, inc=inc, outc=outc)]
            name = 'fusiond'
            latency += self.latency_dict[self.fusion_template.format(
                name=name, inres=outres, outres=outres, inc=outc, outc=outc, fusion=len(conn_list), kernel=kernel)]
        name = 'decoder'
        inres = base_res
        inc = level_channels_dict['level_0']
        latency += self.latency_dict[self.branch_template.format(
            name=name, inres=inres, outres=inres * 2, inc=inc, outc=1)]
        return latency

    def cal_latency(self, inp_res, arch_cfg):
        level_channels_dict = {
            'level_0': arch_cfg['backbone_cfg']['stage_1']['filter'][-1],
            'level_1': arch_cfg['backbone_cfg']['stage_2']['filter'][-1],
            'level_2': arch_cfg['backbone_cfg']['stage_3']['filter'][-1],
            'level_3': arch_cfg['backbone_cfg']['stage_5']['filter'][-1],
            'level_4': arch_cfg['backbone_cfg']['stage_7']['filter'][-1],
        }

        backbone_latency = self.cal_backbone_latency(inp_res, arch_cfg['backbone_cfg'])
        transport_latency = self.cal_latency_transport(inp_res, arch_cfg['transport_cfg'], level_channels_dict)
        decoder_latency = self.cal_latency_decoder(inp_res, arch_cfg['decoder_cfg'], level_channels_dict)
        return {
            'backbone_latency': backbone_latency,
            'transport_latency': transport_latency,
            'decoder_latency': decoder_latency,
            'total_latency': backbone_latency + transport_latency + decoder_latency
        }


if __name__ == '__main__':
    from iNAS.archs.iNAS_supernet.uniform_sampler import Uniform_Architecture_Sampler
    arch = Uniform_Architecture_Sampler().sampling('min')
    calculator = LatencyCalculator(
        'D:\\Codes\\iNAS_DEV-main\\experiments\\pretrained_models\\resource_tables\\latency\\lut_intelcore_cpu.txt',
        device='cpu')
    print(calculator.cal_latency(224, arch))
