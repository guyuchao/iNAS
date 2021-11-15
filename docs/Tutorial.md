# Tutorial

This page provides basic tutorials about the usage of iNAS. Please run the commands in the root path of `iNAS`. <br>
In general, both the training and testing include the following steps:

1. Prepare datasets.
2. Modify config files. The config files are under the `options` folder. For more specific configuration information, please refer to [Config.md](Config.md).
3. You may need to download pre-trained models. Please see [ModelZoo](ModelZoo.md)
4. Run commands. Use [Training Commands](#Training-Commands), [Searching Commands](#Searching-Commands), [Converting Commands](#Converting-Commands) and [Testing Commands](#Testing-Commands) accordingly.

#### Contents

1. [Training Commands](#Training-Commands)
    1. [Single GPU Training](#Single-GPU-Training)
    2. [Distributed (Multi-GPUs) Training](#Distributed-Training)
2. [Searching Commands](#Searching-Commands)
    3. [Benchmark Latency Lookup Table on Devices](#Benchmark-Latency-Lookup-Table-on-Devices)
    4. [Single GPU Searching](#Single-GPU-Searching)
3. [Converting Commands](#Converting-Commands)
4. [Testing Commands](#Testing-Commands)


## Training Commands

### Single GPU Training


> CUDA_VISIBLE_DEVICES=0 python iNAS/train.py -opt [config file]

### Distributed Training

**4 GPUs (default)**

> CUDA_VISIBLE_DEVICES=0,1,2,3
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 iNAS/train.py -opt options/train/SOD/train_supernet_sod_4gpu_b10_e100_noaug.yml --launcher pytorch

## Searching Commands

### Benchmark Latency Lookup Table on Devices

Here, we provide a sample latency lookup table ([Baidu Drive (f3au)](https://pan.baidu.com/s/1SQ-oHt2Kxp1gXxFaChQ9EA) | [Google Drive](https://drive.google.com/drive/folders/1epKOruGhfyfU2ECZniTa2o8VlKPoMl3q?usp=sharing)) tested on Intel Core CPU. You can generate it yourself by following command:

Generate the latency lookup table template (lut_template.txt) based on our search space.
> python scripts/build_latency_table/build_latency_lut_template.py

Use the template file, we can benchmark the component latency on different devices by running the following commands:

> python scripts/build_latency_table/compute_lut_on_devices.py

For mobile phone, this command only generate jit models. We need to benchmark it on mobile phones by [Pytorch Mobile](https://pytorch.org/mobile/home/). Key codes can be found in `scripts/build_latency_table/mobile_scripts`.

### Single GPU Searching

> CUDA_VISIBLE_DEVICES=0 \\\
> python iNAS/search.py -opt options/search/SOD/search_iNAS_1gpu_iter10.yml

## Converting Commands

We can convert the supernet weight for each stand-alone models (initialized by json configuration) using the following command:

> CUDA_VISIBLE_DEVICES=0 \\\
> python iNAS/search.py -opt options/convert/SOD/convert_sod.yml

## Testing Commands

We benchmark the stand-alone model accuracy on different SOD test set by the following command:

> CUDA_VISIBLE_DEVICES=0 \\\
> python iNAS/test.py -opt options/test/SOD/test_standalone_sod.yml
