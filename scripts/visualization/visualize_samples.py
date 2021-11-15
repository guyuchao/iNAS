import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
from operator import itemgetter
from tqdm import tqdm

plt.rc('font', family='Arial')


def update_scatter(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([64])


def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(8)


def read_from_json(json_path):
    with open(json_path, 'r') as fr:
        sample_latency = []
        sample_accuracy = []
        sample_dict = json.load(fr)
        for k, cfg_di in tqdm(sample_dict.items()):
            sample_latency.append(float(cfg_di['CPU']))
            sample_accuracy.append(float(cfg_di['Fmeasure']) * 100)
        xy = list(zip(sample_latency, sample_accuracy))
        sorted_xy = sorted(xy, key=itemgetter(0, 1), reverse=False)
    return sorted_xy


def ParetoTwoDimensions(data):
    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=False)
    assert data == sorted_data
    pareto_idx = list()
    pareto_idx.append(0)
    cut_off = sorted_data[0][1]
    for i in tqdm(range(1, len(sorted_data))):
        if sorted_data[i][1] > cut_off:
            pareto_idx.append(i)
            cut_off = sorted_data[i][1]
    return pareto_idx


if __name__ == '__main__':
    ms_json_path = 'models/population_iter_9.json'
    single_json_path = 'models/population_iter_0.json'

    ss = read_from_json(single_json_path)
    ms = read_from_json(ms_json_path)

    pareto_idx_ss = ParetoTwoDimensions(ss)
    pareto_idx_ms = ParetoTwoDimensions(ms)

    arr_ss = np.array(ss)
    arr_ms = np.array(ms)

    arr_pareto_ss = arr_ss[pareto_idx_ss]
    print(arr_pareto_ss)
    arr_pareto_ms = arr_ms[pareto_idx_ms]

    plt.scatter(arr_ms[::4, 0], arr_ms[::4, 1], s=14, c='#66c2a5', marker='D', label='Random Search', alpha=0.8)
    plt.scatter(arr_ss[::4, 0], arr_ss[::4, 1], s=14, c='#8da0cb', marker='*', label='Hardware Evolution', alpha=0.8)

    plt.plot(arr_pareto_ss[:, 0], arr_pareto_ss[:, 1], c='#8da0cb', markersize=12, marker='*')
    plt.plot(arr_pareto_ms[:, 0], arr_pareto_ms[:, 1], c='#66c2a5', markersize=8, marker='D')
    plt.xlabel('Latency (ms)', fontsize=14)
    plt.ylabel('MaxF', fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.legend(
        handler_map={
            PathCollection: HandlerPathCollection(update_func=update_scatter),
            plt.Line2D: HandlerLine2D(update_func=updateline)
        },
        fontsize=12)
    plt.show()
