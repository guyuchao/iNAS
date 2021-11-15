import argparse
import os.path as osp
from itertools import product

from iNAS.archs.supernet_cfg import SupernetCfg


class Latency_Table_Template:
    """[summary]"""

    def __init__(self):
        super(Latency_Table_Template, self).__init__()
        self.latencylist = []
        self.dynamicmb_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} outc:{outc} '\
            'stride:{stride} kernel:{kernel} ratio:{ratio} latency_mean:{latency_mean} latency_var:{latency_var}\n'
        self.fusion_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} '\
            'outc:{outc} fusion:{fusion} kernel:{kernel} latency_mean:{latency_mean} latency_var:{latency_var}\n'
        self.branch_template = 'name:{name} inres:{inres} outres:{outres} inc:{inc} '\
            'outc:{outc} latency_mean:{latency_mean} latency_var:{latency_var}\n'

        self.level_channels_list = {
            'level_0': SupernetCfg['backbone_cfg']['stage_1']['filter'],
            'level_1': SupernetCfg['backbone_cfg']['stage_2']['filter'],
            'level_2': SupernetCfg['backbone_cfg']['stage_3']['filter'],
            'level_3': SupernetCfg['backbone_cfg']['stage_5']['filter'],
            'level_4': SupernetCfg['backbone_cfg']['stage_7']['filter'],
        }

    def build_latency_table_template(self, inp_res, save_root):
        self.build_backbone(SupernetCfg['backbone_cfg'], inp_res)
        self.build_transport(SupernetCfg['transport_cfg'], inp_res)
        self.build_decoder(SupernetCfg['decoder_cfg'], inp_res)
        with open(osp.join(save_root, 'lut_template.txt'), 'w') as fw:
            fw.writelines(self.latencylist)

    def build_transport(self, cfg, base_res):
        base_res = base_res // 2
        for level_name, level_cfg in cfg.items():
            level_idx = int(level_name.split('_')[1])
            outres = base_res // (2**level_idx)
            connection_list = level_cfg['connection']

            channels = self.level_channels_list[level_name]
            kernel = level_cfg['kernel']
            fusions = list(range(1, len(connection_list) + 1))

            for ckf in product(channels, kernel, fusions):
                c, ks, f = ckf
                name = 'fusiontp'
                self.latencylist.append(
                    self.fusion_template.format(
                        name=name,
                        inres=outres,
                        outres=outres,
                        inc=c,
                        outc=c,
                        fusion=f,
                        kernel=ks,
                        latency_mean='{latency_mean}',
                        latency_var='{latency_var}'))

            for conn_idx in connection_list:
                inres = base_res // (2**conn_idx)
                inc_list = self.level_channels_list['level_%d' % conn_idx]
                for ioc in product(inc_list, channels):
                    inc, outc = ioc
                    name = 'branchtp'
                    self.latencylist.append(
                        self.branch_template.format(
                            name=name,
                            inres=inres,
                            outres=outres,
                            inc=inc,
                            outc=outc,
                            latency_mean='{latency_mean}',
                            latency_var='{latency_var}'))

    def build_decoder(self, cfg, base_res):
        base_res = base_res // 2
        for level_name, level_cfg in cfg.items():
            level_idx = int(level_name.split('_')[1])
            outres = base_res // (2**level_idx)
            connection_list = level_cfg['connection']

            channels = self.level_channels_list[level_name]
            kernel = level_cfg['kernel']
            fusions = list(range(1, len(connection_list) + 1))

            for ckf in product(channels, kernel, fusions):
                c, ks, f = ckf
                name = 'fusiond'
                self.latencylist.append(
                    self.fusion_template.format(
                        name=name,
                        inres=outres,
                        outres=outres,
                        inc=c,
                        outc=c,
                        fusion=f,
                        kernel=ks,
                        latency_mean='{latency_mean}',
                        latency_var='{latency_var}'))
            for conn_idx in connection_list:
                inres = base_res // (2**conn_idx)
                inc_list = self.level_channels_list['level_%d' % conn_idx]
                for ioc in product(inc_list, channels):
                    inc, outc = ioc
                    name = 'branchd'
                    self.latencylist.append(
                        self.branch_template.format(
                            name=name,
                            inres=inres,
                            outres=outres,
                            inc=inc,
                            outc=outc,
                            latency_mean='{latency_mean}',
                            latency_var='{latency_var}'))

        name = 'decoder'
        inres = base_res
        inc_list = self.level_channels_list['level_0']
        for inc in inc_list:
            self.latencylist.append(
                self.branch_template.format(
                    name=name,
                    inres=inres,
                    outres=inres * 2,
                    inc=inc,
                    outc=1,
                    latency_mean='{latency_mean}',
                    latency_var='{latency_var}'))

    def build_backbone(self, cfg, inp_res):
        previous_filters = [3]

        for stage_name, stage_cfg in cfg.items():
            this_filters = stage_cfg['filter']

            depth = stage_cfg['depth']

            if stage_name == 'stem':
                name = 'stem'
            else:
                name = 'dynamicmb'
            stride = stage_cfg['stride']
            kernel = stage_cfg['kernel']
            ratio = stage_cfg['ratio']

            if stride == 2:
                out_res = inp_res // 2
            else:
                out_res = inp_res

            for filter_comb in product(previous_filters, this_filters, kernel):
                inc, outc, ks = filter_comb
                self.latencylist.append(
                    self.dynamicmb_template.format(
                        name=name,
                        inres=inp_res,
                        outres=out_res,
                        inc=inc,
                        outc=outc,
                        kernel=ks,
                        stride=stride,
                        ratio=ratio,
                        latency_mean='{latency_mean}',
                        latency_var='{latency_var}'))

            inp_res = out_res
            previous_filters = this_filters
            stride = 1

            if max(depth) > 1:
                for filter_comb in product(previous_filters, this_filters, kernel):
                    inc, outc, ks = filter_comb
                    self.latencylist.append(
                        self.dynamicmb_template.format(
                            name=name,
                            inres=inp_res,
                            outres=out_res,
                            inc=inc,
                            outc=outc,
                            kernel=ks,
                            stride=stride,
                            ratio=ratio,
                            latency_mean='{latency_mean}',
                            latency_var='{latency_var}'))
        return self.latencylist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=224, help='input resolution')
    parser.add_argument(
        '--save_root',
        type=str,
        default='experiments/pretrained_models/resource_tables/latency/',
        help='Save root path.')
    args = parser.parse_args()
    Latency_Table_Template().build_latency_table_template(inp_res=args.res, save_root=args.save_root)
