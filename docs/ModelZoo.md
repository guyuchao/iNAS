# Model Zoo

We provide:

1. Official trained supernet models and logs.
2. Searched models on different devices, their converted weights and predicted saliency maps.

You can put the downloaded models in the `experiments/pretrained_models/` folder.

### Trained supernet

| iNAS Supernet | training iteration | horizontal flip | ECSSD Fm@biggest | ECSSD Fm@smallest | download    |
|:--------------:|:----------------:|:----:|:----:|:----:|:----:|
| paper | 26400 (100 epoch)  | No  | 0.948 | 0.940 |  |
| iNAS_SOD_100e_noaug | 26400 (100 epoch)  | No  |  0.952  |  0.942  | [Google Drive](https://drive.google.com/drive/folders/1rKzV2u4VDdydk24Yv8dfnxkQ9OErduPy?usp=sharing) / [Baidu Drive (11tn)](https://pan.baidu.com/s/1kUPj8e2gfI5uekxRCbjVFQ)|
| iNAS_SOD_100e_aug   | 26400 (100 epoch)  | Yes | 0.954 | 0.945 | [Google Drive](https://drive.google.com/drive/folders/1rKzV2u4VDdydk24Yv8dfnxkQ9OErduPy?usp=sharing) / [Baidu Drive (5ktg)](https://pan.baidu.com/s/1forUEvr2QO302G-KDUpuaA)|
| iNAS_SOD_200e_aug   | 52700 (200 epoch)  | Yes | 0.955 |  0.948  | [Google Drive](https://drive.google.com/drive/folders/1ggl3JXQ1GEqbwRuqDFbw-iAAXvWBHPAG?usp=sharing) / [Baidu Drive (vp8q)](https://pan.baidu.com/s/1XuHwyIqG74xfvVSxHqGZBA)|

### Searched models on Intel Core CPU

<table>
<tr>
    <th>iNAS Specialized Sub-nets </th>
    <th>ECSSD Fm/MAE</th>
    <th>#Params</th>
    <th>#Flops</th>
    <th>#Latency@CPU</th>
    <th>Download</th>
</tr>
<tr>
    <td style="text-align:center; font-weight:bold"> Handcraft Models </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
</tr>
<tr>
    <td style="text-align:center"> EGNet-R (ICCV19) </td>
    <td style="text-align:center"> 0.947/0.037 </td>
    <td style="text-align:center"> 111.64M </td>
    <td style="text-align:center"> 120.85G </td>
    <td style="text-align:center"> 791.95ms </td>
    <td rowspan="4">We provide a speed benchmark of handcraft models and you can download it from <a href="https://drive.google.com/file/d/1Snb0tCFxbZZS8nWuKOuNx_G3wI7w8m-G/view?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1ywKVEgtXyn_6l81ExyixeA">Baidu Drive (b34c)</a>.</td>
</tr>
<tr>
    <td style="text-align:center"> ITSD-R (CVPR20) </td>
    <td style="text-align:center"> 0.947/0.034 </td>
    <td style="text-align:center"> 26.47M </td>
    <td style="text-align:center"> 9.65G </td>
    <td style="text-align:center"> 151.51ms </td>
</tr>
<tr>
    <td style="text-align:center"> CSNet (ECCV20) </td>
    <td style="text-align:center"> 0.916/0.065 </td>
    <td style="text-align:center"> 0.14M </td>
    <td style="text-align:center"> 0.72G </td>
    <td style="text-align:center"> 131.11ms</td>
</tr>
<tr>
    <td style="text-align:center"> U2-Net (PR20) </td>
    <td style="text-align:center"> 0.943/0.041 </td>
    <td style="text-align:center"> 1.13M </td>
    <td style="text-align:center"> 9.77G </td>
    <td style="text-align:center"> 186.53ms </td>
</tr>
<tr>
    <td style="text-align:center; font-weight:bold"> Supernet: iNAS_SOD_100e_noaug</td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td rowspan="5">Models and logs can be downloaded from <a href="https://drive.google.com/drive/folders/1DjB9xPDMwAnrQQHoc2RDVG7grTQ67C6J?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1ForIlqNJj7YDqDRR9SC3gg">Baidu Drive (jgg3).</a></td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@27.00ms_Fmeasure@0.9437</td>
    <td style="text-align:center">0.943/0.036</td>
    <td style="text-align:center">5.09M</td>
    <td style="text-align:center">0.45G</td>
    <td style="text-align:center">29.97ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@30.42ms_Fmeasure@0.9462</td>
    <td style="text-align:center">0.946/0.034</td>
    <td style="text-align:center">5.83M</td>
    <td style="text-align:center">0.58G</td>
    <td style="text-align:center">33.03ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@35.76ms_Fmeasure@0.9493</td>
    <td style="text-align:center">0.949/0.034</td>
    <td style="text-align:center">8.15M</td>
    <td style="text-align:center">0.69G</td>
    <td style="text-align:center">38.74ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@45.55ms_Fmeasure@0.9522</td>
    <td style="text-align:center">0.952/0.031</td>
    <td style="text-align:center">13.44M</td>
    <td style="text-align:center">0.85G</td>
    <td style="text-align:center">49.26ms</td>
</tr>
<tr>
    <td style="text-align:center; font-weight:bold"> Supernet: iNAS_SOD_200e_aug</td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td style="text-align:center"> </td>
    <td rowspan="5">Models and logs can be download from <a href="https://drive.google.com/drive/folders/1epKOruGhfyfU2ECZniTa2o8VlKPoMl3q?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1jz9_X-p4M7jF6nC3hY_6yw">Baidu Drive (x60e)</a>.</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@26.99ms_Fmeasure@0.9487</td>
    <td style="text-align:center">0.949/0.034</td>
    <td style="text-align:center">5.15M</td>
    <td style="text-align:center">0.48G</td>
    <td style="text-align:center">29.08ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@34.80ms_Fmeasure@0.9520</td>
    <td style="text-align:center">0.952/0.032</td>
    <td style="text-align:center">7.48M</td>
    <td style="text-align:center">0.65G</td>
    <td style="text-align:center">37.35ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@40.84ms_Fmeasure@0.9540</td>
    <td style="text-align:center">0.954/0.031</td>
    <td style="text-align:center">9.23M</td>
    <td style="text-align:center">0.75G</td>
    <td style="text-align:center">44.20ms</td>
</tr>
<tr>
    <td style="text-align:center">CPU_lat@59.10ms_Fmeasure@0.9560</td>
    <td style="text-align:center">0.956/0.030</td>
    <td style="text-align:center">16.29M</td>
    <td style="text-align:center">1.08G</td>
    <td style="text-align:center">61.88ms</td>
</tr>
</table>

### Searched models on Mobile Phone: coming soon
### Searched models on GPU: coming soon
### Searched models on Embedded Device: coming soon
