# iNAS: Integral NAS for Device-Aware Salient Object Detection

## Introduction

**Integral search design (jointly consider backbone/head structures, design/deploy devices).**

---

<div align="center">
  <img src="assets/SepDesign.PNG" width="330" height="250"/> <img src="assets/SearchDesign.PNG" width="350" height="250"/>
</div>


**Covers mainstream handcraft saliency head design.**
<div align="center">
  <img src="assets/cover.PNG" width="500" height="200"/>
</div>

**SOTA performance with large latency reduction on diverse hardware platforms.**

---

<div align="center">
  <img src="assets/bench1.PNG" width="500"/>
</div>

<div align="center">
  <img src="assets/bench2.PNG" width="500"/>
</div>

## Updates

**0.1.0** was released in 15/11/2021:
- Support training and searching on Salient Object Detection (SOD).
- Support four stages in one-shot architecture search.
- Support stand-alone model inference with json configuration.
- Provide off-the-shelf models and experiment logs.

Please refer to [changelog.md](docs/changelog.md) for details and release history.


## Dependencies and Installation

**Dependencies**

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

**Install from a local clone**

1. Clone the repo

    ```bash
    git clone https://github.com/guyuchao/iNAS.git
    ```

2. Install dependent packages

    ```bash
    conda create -n iNAS python=3.8
    conda install -c pytorch pytorch=1.7 torchvision cudatoolkit=10.2
    pip install -r requirements.txt
    ```

3. Install iNAS<br>
    Please run the following commands in the **iNAS root path** to install iNAS:<br>

    ```bash
    python setup.py develop
    ```

## Dataset Preparation

**Folder Structure**

```
iNAS
├── iNAS
├── experiment
├── scripts
├── options
├── datasets
│   ├── saliency
│   │   ├── DUTS-TR/            # Contains both images (.jpg) and labels (.png).
│   │   ├── DUTS-TR.lst         # Specify the image-label pair for training or testing.
│   │   ├── ECSSD/
│   │   ├── ECSSD.lst
│   │   ├── ...
```
**Common Image SOD Datasets**

We provide a list of common salient object detection datasets.
<table>
<tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Download</th>
</tr>
<tr>
    <td rowspan="1">SOD Training</td>
    <td>DUTS-TR</td>
    <td><sub>10553 images for SOD training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/file/d/1Put-5roLAwuGU9gJBdY8fMEZKC9XjJfH/view?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1qxCgh4sFV_KnRQs1jMQ7hQ">Baidu Drive (psd: w69q)</a></td>
</tr>
<tr>
    <td rowspan="5">SOD Testing</td>
    <td>ECSSD</td>
    <td><sub>1000 images for SOD testing</sub></td>
</tr>
<tr>
    <td>DUT-OMRON</td>
    <td><sub>5168 images for SOD testing</sub></td>
</tr>
<tr>
    <td>DUTS-TE</td>
    <td><sub>5019 images for SOD testing</sub></td>
</tr>
<tr>
    <td>HKU-IS</td>
    <td><sub>4447 images for SOD testing</sub></td>
</tr>
<tr>
    <td>PASCAL-S</td>
    <td><sub>850 images for SOD testing</sub></td>
</tr>
</table>


## How to Use

The iNAS integrates four main steps of one-shot neural architecture search:
- Train supernet: Provide a fast performance evaluator for searching.
- Search models: Find a pareto frontier based on performance evaluator and resource evaluator.
- Convert weight/Retrain/Finetune: Promote searched model performance to its best. (We now support converting supernet weight to stand-alone models without retraining.)
- Deploy: Test stand-alone models.

Please see [Tutorial.md](docs/Tutorial.md) for the basic usage of those steps in iNAS.

## Model Zoo

Pre-trained models and log examples are available in [ModelZoo.md](docs/ModelZoo.md).

## TODO List

- [ ] Support multi-processing search (simply use data-parallel cannot increase search speed).
- [ ] Complete documentations.
- [ ] Add some applications.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@inproceedings{gu2021inas,
  title={iNAS: Integral NAS for Device-Aware Salient Object Detection},
  author={Gu, Yu-Chao and Gao, Shang-Hua and Cao, Xu-Sheng and Du, Peng and Lu, Shao-Ping and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4934--4944},
  year={2021}
}
```

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (cc-by-nc-sa)](https://creativecommons.org/licenses/by-nc-sa/4.0/), where only
non-commercial usage is allowed. For commercial usage, please contact us.

## Acknowledgement
The project structure is borrowed from [BasicSR](https://github.com/xinntao/BasicSR), and parts of implementation and evaluation codes are borrowed from [Once-For-All](https://github.com/mit-han-lab/once-for-all), [BASNet](https://github.com/xuebinqin/BASNet) and [BiSeNet
](https://github.com/CoinCheung/BiSeNet). Thanks for these excellent projects.

## Contact
If you have any questions, please email `ycgu@mail.nankai.edu.cn`.
