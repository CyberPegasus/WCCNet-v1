# WCCNet

> [WCCNet: Wavelet-integrated CNN with Crossmodal Rearranging Fusion for Fast Multispectral Pedestrian Detection](https://arxiv.org/abs/2308.01042)

This repository contains the supported code, configuration files and model weights of [our proposed WCCNet](https://arxiv.org/abs/2308.01042). 

The contents are as follows:

- [Introduction](#introduction)
- [Usage](#usage)
  * [1. Environment Settings](#1-environment-settings)
    + [1.1 Python Dependencies](#11-python-dependencies)
    + [1.2 Installation](#12-installation)
  * [2. Dataset Preparation](#2-dataset-preparation)
    + [2.1 Pre-Steps](#21-pre-steps)
    + [2.2 KAIST Preparation](#22-kaist-preparation)
    + [2.3 FLIR Preparation](#23-flir-preparation)
  * [3. Evaluation](#3-evaluation)
    + [3.1 Inference on Datasets for Prediction Results](#31-inference-on-datasets-for-prediction-results)
    + [3.2 FLIR Evaluation](#32-flir-evaluation)
    + [3.3 KAIST Evaluation](#33-kaist-evaluation)
  * [4. Training](#4-training)
    + [4.1 Configuration Selection](#41-configuration-selection)
    + [4.2 Run with single GPU](#42-run-with-single-gpu)
    + [4.3 Run with multi GPUs on one device](#43-run-with-multi-gpus-on-one-device)
  * [5. Brief Illustration on Customized Models or Customized Datasets](#5-brief-illustration-on-customized-models-or-customized-datasets)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction

For efficient multispectral pedestrian detection, we propose a novel wavelet-integrated CNN framework named WCCNet that is able to differentially extract rich features of different spectra with lower computational complexity and semantically rearranges these features for effective crossmodal fusion. For more details, please refer to [our paper](https://arxiv.org/abs/2308.01042).



<h2 align="center">Log-average Miss Rate vs Inference Time Comparisons on KAIST</h2>

<img align="center" src="assets/kaist_log2yx.png" width="1000">

<h2 align="center">Comparisons on KAIST with All-dataset (left) and Reasonable (right) Settings</h2>

<img src="assets/kaist_MR2Curve.png" width="1000">

## Usage
### 1. Environment Settings

> We train WCCNet on **RTX 3090**, and test it on a series NVIDIA GPUs, including **TITAN X, GTX 1080TI, RTX 2060s, and RTX 3090**.
> 
> WCCNet is implemented with the environment settings as follows: **$\left[ \mathrm{Ubuntu}=18.04, \mathrm{Python}=3.7, \mathrm{PyTorch}=1.8.2, \mathrm{cudatoolkit}=10.2 \right]$**.
> 
> VSCode is adopted as our IDE.
> 
> It is recommended to use our environment settings.

#### 1.1 Python Dependencies

- Configure Python environment with conda

  ```bash
  # create conda environment
  conda create -n wccnet python=3.7
  conda activate wccnet
  # install pytorch==1.8.2 LTS
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
  # install pytorch_wavelets
  git clone https://github.com/fbcotter/pytorch_wavelets
  cd pytorch_wavelets
  pip install .
  cd ..
  # install other python dependencies
  pip install -r requirements.txt
  ```

- Manually configure Python environment

  - For manual configuration of python environment, detailed requirements are provided in `./requirements.txt` in this repository.
  - Version requirements
    - Python >= 3.7
    - PyTorch >=1.7, while PyTorch==1.8.2 LTS is suggested
    - cudatoolkit >= 10.1
    - pycocotools >= 2.0.2

#### 1.2 Installation

- Install WCCNet from source

  ```bash
  git clone https://github.com/CyberPegasus/WCCNet.git
  cd WCCNet # Note that this path refers to the subsequent {the root dir of WCCNet}
  pip install -v -e .
  # "-v" means verbose, or more output,
  # "-e" means installing a project in editable mode to enable local modifications on-the-fly.
  # "." means referring to the current dir
  ```



### 2. Dataset Preparation

For multispectral pedestrian detection, we train and test WCCNet on the KAIST and FLIR dataset, which should be first downloaded. These datasets should be stored in `./datasets/` and renamed to `kaist` and `flir` respectively. The detailed information is provided below.

#### 2.1 Pre-Steps

- **File hierarchy organization**

  Organize the file hierarchy as follows:

  ```bash
  mkdir datasets
  cd datasets
  mkdir kaist
  mkdir alignedFLIR
  mkdir scripts
  cd scripts
  mkdir KAIST
  mkdir FLIR
  cd ..
  ```

- **Annotation Script Preparation**

  We provide the annotation generation scripts for KAIST and FLIR datasets, which will convert their original annotations into the COCO form.  You are expected to download these scripts in [kaist_link](https://github.com/CyberPegasus/storage/tree/master/lightweight_files/scripts/KAIST) and [flir_link](https://github.com/CyberPegasus/storage/tree/master/lightweight_files/scripts/FLIR), and unzip their contents to `./datasets/scripts/KAIST/` and `./datasets/scripts/FLIR/` respectively.

#### 2.2 KAIST Preparation

- Download KAIST to `./datasets/kaist/` and unzip

  - Manual way
    1. Visit [KAIST repos](https://github.com/SoonminHwang/rgbt-ped-detection/)
    2. Download the [KAIST dataset](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data) (Raw links are shown below)
       - Train
         - [set00](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set00.zip), [set01](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set01.zip), [set02](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set02.zip), [set03](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set03.zip), [set04](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set04.zip), [set05](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set05.zip)
       - Test
         - [set06](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set06.zip), [set07](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set07.zip), [set08](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set08.zip), [set09](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set09.zip), [set10](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set10.zip), [set11](http://multispectral.kaist.ac.kr/pedestrian/data-kaist/images/set11.zip)
  - Third-party way: Download the decoded KAIST dataset in the form of `.png` images (Suggested)
    - The authors of [MBNet](https://arxiv.org/pdf/2008.03043.pdf) provide the decoded train and test data of KAIST dataset.
    - The corresponding netdisk links are shown in the [Prerequisites](https://github.com/CalayZhou/MBNet#2-prerequisites) of [MBNet repos](https://github.com/CalayZhou/MBNet).
    - Our provided script to generate labels of testing set is designed for the data downloaded from this third-party way.
 
- Prepare original annotations

    - Following common annotation settings for a fair comparison with recent related works, the [paired annotations](https://github.com/luzhang16/AR-CNN) in [ARCNN](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.html) are adopted for training and the [cleaned annotations](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) provided by [Liu et al.](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn) are applied for testing.  Notice that the testing part (set06~set11) of the paired annotations is exactly equal to the cleaned annotations by Liu et al. In other words, if we calculate and round up the mean values of the RGB / IR bounding boxes in the paired annotations, then we obtain the cleaned annotations. This is implemented in our provided scripts.
    - Therefore, download the [paired annotations](https://github.com/luzhang16/AR-CNN), and unzip all the contents in `kaist_paired/annotations/*` to `./datasets/kaist/annos/paired/`

- File organize

    - Now we assume you download the KAIST through the Third-party way

    - Create folders for train, validation and test set 

        ```bash
        cd datasets/kaist
        mkdir train # store the downloaded set00~set05 directory
        mkdir test # store the visible and lwir directory downloaded from the third-party way
        ln -s train val # create a softlink for validation folder
        mkdir split
        mkdir annotations # store the generated label
        mkdir annos # store the original label
        ```

        The specifically used images of training and validation set are controlled by the generated train and val annotations in the next step. 

    - The downloaded KAIST train set and test set should be moved to `train` and `test` folder respectively. 

- Using provided files for train/test splitting

    - Download [trainval.txt](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/kaist/trainval.txt), [text.txt](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/kaist/test.txt), and [split_trainval.py](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/kaist/split_trainval.py) to `./datasets/kaist/split/`

    - Go into `./dataset/kaist/split/` and run `split_trainval.py`

      ```bash
      cd datasets/kaist/split/
      python split_trainval.py
      ```

      Then the train.txt and val.txt are expected to be generated.

- Generate COCO-like annotations
    - fix broken image: `datasets/kaist/test/visible/set10_V000_I02979_visible.png` is broken, substitute `datasets/kaist/test/visible/set10_V000_I02959_visible.png` for it.

    - Run the scripts `kaist2coco_train.py`, `kaist2coco_val.py` and `kaist2coco_test.py` provided in [link](https://github.com/CyberPegasus/storage/tree/master/lightweight_files/scripts/KAIST).

        ```bash
        cd {the root dir of WCCNet}
        python datasets/scripts/KAIST/kaist2coco_train.py
        python datasets/scripts/KAIST/kaist2coco_val.py
        python datasets/scripts/KAIST/kaist2coco_test.py
        ```

    - Then we will get the corresponding annotations in the form of COCO


#### 2.3 FLIR Preparation

- Download FLIR and Unzip

  - Following common annotation settings, aligned FLIR by Zhang et al. is used for training and evaluation. It removes images that do not have the counterparts in the corresponding modality, and retains only the three most commonly used categories including humans, cars, and bicycles. Since the original link provided by its authors does not work, it now can be downloaded from [link](https://drive.google.com/file/d/1xHDMGl6HJZwtarNWkEV3T4O9X4ZQYz2Y/view) in this [issue](https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource/issues/6).

  - File organize

    ```bash
    cd datasets/alignedFLIR
    mkdir annos
    mkdir annotations
    mkdir train
    mkdir test
    mkdir val
    ```

  - Unzip the contents in `align.zip/align/` to `./datasets/alignedFLIR/`

    - Rename the `Annotations/` directory to `Annotations_origin/`

- Prepare annotations

  - Copy and rename annotations

    ```bash
    cp Annotations_origin/*.xml annos/
    cd annos/
    rename 's/FLIR_//' *.xml
    rename 's/_PreviewData//' *.xml
    cd ..
    ```

  - Run the preprocessing script [`flirAnno_preprocess.py`](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/scripts/FLIR/flirAnno_preprocess.py) in `./datasets/scripts/FLIR/`

    ```bash
    cd {the root dir of WCCNet}
    python datasets/scripts/FLIR/flirAnno_preprocess.py
    ```

- Generate COCO-like annotations

  - Create softlinks for train, val, and test folder for convenience.

    ```bash
    cd datasets/alignedFLIR/
    ln -s JPEGImages train
    ln -s JPEGImages val
    ln -s JPEGImages test
    ```
  
    > **Note that** this softlink operation **does not** mean the training, validation and testing set using the same sample !
  >
    > The **specifically used images** for different sets are **controlled by the generated train, val and test annotations in the next step**.
  
  - Run the script [`FLIR_voc2coco.py`](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/scripts/FLIR/FLIR_voc2coco.py) in `./datasets/scripts/FLIR/` to generate annotations for train, val and test.
  
    ```bash
    cd {the root dir of WCCNet}
    python datasets/scripts/FLIR/FLIR_voc2coco.py
    ```
  

### 3. Evaluation

#### 3.1 Inference on Datasets for Prediction Results

Generate predictions for specific model on KAIST or FLIR dataset.

> **Notes**: The prediction results of WCCNet are provided in `./outputs/predictions/`. It can be reproduced follow the steps that introduced bellow.

- Download provided `.pth` model weights to `./weights/`

  - WCCNet for KAIST: [weights link](https://github.com/CyberPegasus/storage/releases/download/v1.0.0/wccnet_kaist.pth)

  - WCCNet for FLIR: [weights link](https://github.com/CyberPegasus/storage/releases/download/v1.0.0/wccnet_flir.pth)
 
- Configuration

  - Choose one corresponding `.json` configuration file for a downloaded model weight. The corresponding configuration files are stored in `./cfg/`.

  - You need first to configure the configuration file. Taking `cfg/kaist/WCCNet_kaist.json` as example:

    1. `experiment_name`: The output folder name in `./exps_results/` recording necessary messages and weights. It also controls the name of the output prediction files
    2. `batch_size`: Set the batch size during evaluation process based on your GPU memory size. 

  - Set the string value of `ckpt` in the configuration file to the **path** of your downloaded model weight. Absolute path is suggested. 

- Run the evaluation code `./tools/eval_singleGPU.py` with the selected configuration file

  ```bash
  cd {the root dir of WCCNet}
  python tools/eval_singleGPU.py -cfg {the relative path of the selected configuration file}
  # take cfg/kaist/WCCNet_kaist.json as example
  # python tools/eval_singleGPU.py -cfg cfg/kaist/WCCNet_kaist.json
  ```

  - Then the parameters and the computational cost will be shown in the command console and written to the log files stored in `./exps_results/{expriment_name}/`. 
  - The predictions will be stored in `./outputs/predictions/` in the form of `.json` file.
  - Inference time notification: **The inference time per sample in evaluation process** containing only one epoch **will be longer than** **the one in training process** containing more epochs, due the well-known **warmup process of CUDA** including cudaMalloc, etc. Besides, **the inference time also depends on the performance of your applied CPU**.

- Another way: Use the weight reproduced from our provided training tools

  - After training WCCNet models via Step 4. Training, the model weights of the latest checkpoints and the best checkpoints on validation set will be stored in `./exps_results/{expriment_name}/`.

  - The `expriment_name` is defined in a specific `.json` configuration file under `./cfg/`.

  - The best checkpoints on validation set will be automatically used if `ckpt` set to `null` in your configuration file.

  - Run the evaluation code `./tools/eval_singleGPU.py` with the selected configuration file

    ```bash
    cd {the root dir of WCCNet}
    python tools/eval_singleGPU.py -cfg {the relative path of the selected configuration file}
    # take cfg/kaist/WCCNet_kaist.json as example
    # python tools/eval_singleGPU.py -cfg cfg/kaist/WCCNet_kaist.json
    ```

#### 3.2 FLIR Evaluation

For FLIR dataset, following steps in **3.1 Inference on Datasets for Prediction Results**, the mAP and AP of each class have already been printed in command console, and be recorded to log files stored in `./exps_results/{expriment_name}/`.

#### 3.3 KAIST Evaluation

For KAIST dataset, we further generate additional evaluation results in the form of $\mathrm{MR}^{-2}$ metrics and corresponding comparison log-average MR curves.

- Follow the above-mentioned steps, now prediction result files are generated and stored in `./outputs/predictions/`, 

- Firstly, run the convert script `./outputs/predictions/json2txt.py` to convert the  `.json` file to `.txt` file. It is also provided in [this link](https://github.com/CyberPegasus/storage/blob/master/lightweight_files/scripts/evaluation/json2txt.py).

  ```bash
  cd {the root dir of WCCNet}
  python outputs/predictions/json2txt.py -res {the relative path of the prediction result file}
  # take outputs/predictions/WCCNet_kaist.json as example
  # python outputs/predictions/json2txt.py -res outputs/predictions/WCCNet_kaist.json
  ```

- Secondly, copy the generated `.txt` file to `./evaluation/kaist/`. Note that the result files of other prior works are also provided in `./evaluation/kaist`, which are provided by [MLPD](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection). And the results of WCCNet are provided in `./evaluation/kaist/WCCNets/`.

- Thirdly, give result files to be compared and evaluated in `evaluation/evaluation_script_kaist.py`

  - Change `rstFiles` string list to select result files, input their relative path.

- Lastly, run `./evaluation/evaluation_script_kaist.py` for evaluation results

  ```bash
  cd {the root dir of WCCNet}
  python evaluation/evaluation_script_kaist.py
  ```

  - The output evaluation results and corresponding log-average MR curves will be stored in `./evaluation/`.

### 4. Training

#### 4.1 Configuration Selection

Choose configuration files and experiment files for specific model on KAIST or FLIR dataset

- We use configuration files in `./cfg/` to select model for training process. Besides, the training hyperparameters, model definition, and dataset settings are defined in `.py` experiment files in `./exps/user/`. The selected experiment file should correspond to the selected configuration file, which means you should specify the applied experiment files via the value of key `"exp_file"` in the corresponding configuration file. For example, set `"exp_file": "exps/user/kaist/WCCNet.py"` for `cfg/kaist/WCCNet_kaist.json`.
- Note: How to train our variants of WCCNet including the models shown in ablation study of our paper
  
  - Different numbers of parameters
    - Controlled by the variable `depth_scale` in corresponding `.py` experiment file in `./exps/user/`.
    - `depth_scale` refers to $\tau$ in our paper:
      - `depth_scale=1.0`: WCCNet-L
      - `depth_scale=0.5`: WCCNet
      - `depth_scale=0.25`: WCCNet-S
      - `depth_scale=0.125`: WCCNet-XS
    - Smaller model means lower computational cost but greater underfitting risk, resulting in faster inference speed but lower accuracy. 
  
  - Backbone of different types
    - The backbone types are specified in corresponding `.py` experiment file in `./exps/user/`. We can adopt a specific type by change the input string value of `Backbone` for `backbone = WCCNet_backbone(depth=33, Backbone='xxx',...)` in this experiment file.

  - Different Fusion Schemes
    - The applied fusion schemes are also specified by the input string value of `Backbone` for `backbone = WCCNet_backbone(depth=33, Backbone='xxx',...)` in the corresponding experiment file.

#### 4.2 Run with single GPU

- Specify the used configuration file by inputting relative path to `cfg_file =xxx/xxx.json` in `tools/train_singleGPU.py`, e.g., `cfg_file = cfg/flir/WCCNet_flir.json`

- Specify the experiment name by `"experiment_name"` in the configuration file.

- Specify the used `GPU ID` by `"gpuid"` in the configuration file.

- Run `./tools/train_singleGPU.py`

```bash
cd {the root dir of WCCNet}
python tools/train_singleGPU.py
```

- The training logs and saved model weights will be stored in `./exps_results/{experiment_name}/`

#### 4.3 Run with multi GPUs on one device

- Specify the used configuration file by inputting relative path to `cfg_file =xxx/xxx.json` in `tools/train_singleGPU.py`, e.g., `cfg_file = cfg/flir/WCCNet_flir.json`

- Specify the experiment name by `"experiment_name"` in the configuration file.

- Specify the used GPU number by `"devices"` in the configuration file, and set `"multigpu"` to `True`.

- Specify the used GPU IDs by setting `CUDA_VISIBLE_DEVICES` at the head of`./tools/train_multiGPU.py`

For example:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,5" # Only used two GPUs with ID={4,5}
```

- Run `./tools/train_multiGPU.py`

```bash
cd {the root dir of WCCNet}
python tools/train_multiGPU.py
```

- The training logs and saved model weights will be stored in `./exps_results/{experiment_name}/`

### 5. Brief Illustration on Customized Models or Customized Datasets

1. Add your own backbones, detection heads, and other new modules to `./wccnet/models/`
2. Add your experiment files to `./exps/` to form your customized modules into an integrated network and set corresponding hyperparameters.
3. Add your configuration files to `./cfg/` for training and evaluation.
4. For customized datasets, please change the behaviors of `dataloader` and `PyTorch-like dataset` in `./wccnet/data/` and `./wccnet/exp/yolox_base.py`. Sometimes you need to convert your own annotations into COCO form, which could refer to our provided annotation scripts in **2.1 Pre-Steps**.
5. Use our provided training and evaluation tools to train and evaluate your own models.

## Acknowledgement

We appreciate following providers.

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)
- [log-average MR evaluation](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection).

## Citation

If you use WCCNet in your research, please cite our work:
```latex
@article{wccnet2023,
	title = {WCCNet: Wavelet-integrated CNN with Crossmodal Rearranging Fusion for Fast Multispectral Pedestrian Detection},
	author = {Wang, Xingjian and Chai, Li and Chen, Jiming and Shi, Zhiguo},
	journal={arXiv preprint arXiv:2308.01042},
	year={2023}
}
```
