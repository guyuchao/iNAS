# Put pre-trained models here

#### Download official pre-trained models and pre-computed resource tables.

[Baidu Drive](https://pan.baidu.com/s/1SQ-oHt2Kxp1gXxFaChQ9EA) (Password: f3au) | [Google Drive](https://drive.google.com/drive/folders/1aF_ELGdaD-3u5y2HMylzPGuGumAQ7lbE?usp=sharing)

If you need searched models, please refer to [ModelZoo.md](../docs/ModelZoo.md)


```
iNAS
├── iNAS
├── datasets
├── scripts
├── options
├── experiment/pretrained_models
│   ├── search_space
│   │   ├── checkpoint-28b11d7f.pth                             # Imagenet pretrained supernet.
│   │   ├── train_supernet_sod_4gpu_b10_e100_noaug/             # Put supernet training exp-dir here.
│   │   ├── ...
│   ├── resource_tables/latency                                 # Show latency example only, but you can build other tables, like Flops, parameters.
│   │   ├── lut_template.txt                                    # Template lookup table based on iNAS search space.
│   │   ├── lut_intelcore_cpu.txt                               # Latency lookup table example on Intel Core CPU, you can benchmark on other devices.
│   │   ├── ...
│   ├── searched_models/search_iNAS_1gpu_iter10/                # Put searching exp-dir here.
│   │   ├── models/CPU_search                                   # A series of searched models on pareto frontier.
│   │   ├── models/population_iter_x.json                       # Json checkpoint of searched results.
│   │   ├── train_search_iNAS_1gpu_iter10_xxx.log
```
