import argparse

from iNAS.archs.iNAS_standalone_arch import iNASStandalone
from iNAS.utils.resource_benchmark_util import EvalSpeed, get_net_Gflops_Mparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_config', type=str, default='CPU_search/CPU_lat@27.00ms_Fmeasure@0.9437.json')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    speed_evaluator = EvalSpeed(args.device, args.image_size)
    model = iNASStandalone(args.model_config, deploy=True)

    resource_result = get_net_Gflops_Mparams(model, 224)
    print(resource_result)

    time_spent = speed_evaluator.eval(model)
    print('Avg execution time (ms): %.4f, FPS:%d' % (time_spent, 1000 / time_spent))
