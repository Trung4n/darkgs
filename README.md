## DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark

> "Even with these dark eyes, a gift of the dark night, I go to seek the shining light."   --Gu Cheng 1956-1993

<p align="center">
    <img src="cmu_ri_logo.png" alt="Logo" width="40%"">   
    <img src="NOAA_logo_mobile.svg" alt="Logo" width="25%">
  </a>
</p>

### Novel-view rendering: Simulating a light cone and re-illuminating the environment.
Please check our [videos](https://www.linkedin.com/posts/tianyi-zhang-396b0a186_darkgs-building-3d-gaussians-with-a-torch-activity-7197672371393019905-iY2-?utm_source=share&utm_medium=member_desktop) ([Bilibili](https://www.bilibili.com/video/BV1Euu4eqEtN/?vd_source=ccc6b1a36055375ca812070948900795#reply222119730496)).
<p align="center">
    <img src="darkgs.gif" alt="Logo" width="100%">
  </a>
</p>

### Sister Repo for Camera-Light calibration
The sister repo Neural Light Simulator for light-camera calibration is [here](https://github.com/tyz1030/neuralight). 

## Install
Installation generally follows vanilla Gaussian Splatting installation.
```
git clone git@github.com:tyz1030/darkgs.git --recursive
```
or
```
git clone https://github.com/tyz1030/darkgs.git --recursive
```
Conda environment setup
```
conda env create --file environment.yml
conda activate darkgs
```
Also need to install lietorch
```
pip install git+https://github.com/princeton-vl/lietorch.git
```


## Data
Please find our example data on [Google Drive](https://drive.google.com/drive/folders/1EzhrEBCEHCSF3jtRwMXQpqF9wgh4KlPD?usp=drive_link) and [DropBox](https://www.dropbox.com/scl/fo/nc61inva76a40u934iit0/AAywA7NXF1adODJRnJT2gJI?rlkey=bw4p3ut569ngiml6x5o286zp5&st=jgv98hvj&dl=0).
#### Make your own data
Please put your RAW images subfolder named "raw". To make COLMAP less struggle, I gamma-curved/manually increased the brightness of the raw images for feature extraction and matching. These corrected images are put in "input" subfolder. We only use "raw" images to build DarkGS.
```
python3 convert.py -s <path to your own dataset>
```
## Light Calibration
If you are using your own light-camera setup, please calibrate your system using [neural light simulator](https://github.com/tyz1030/neuralight). Then put model_parameters.pth in the root directory. 

## Quick Start
Train
```
python train.py -s <path to example dataset>
```
Train with [Weights & Biases](https://wandb.ai) logging (off by default, same behaviour as MonoGS: when the flag is absent the run is created with `mode="disabled"`)
```
python train.py -s <path to example dataset> --use_wandb
```
Optional flags: `--wandb_project` (default `DarkGS`), `--wandb_name`, `--wandb_log_interval` (default `10` iterations).\
On Kaggle/Colab (no interactive login) provide your API key through the environment before training, e.g. with Kaggle Secrets:
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```
Train on your own calibrated / simulated data (three optional flags, defaults keep the original behaviour):
```
python train.py -s <dataset> --linearize --scaling_factor 1.0 --light_params <dataset>/model_parameters.pth
```
- `--linearize`: the input images are sRGB-encoded (e.g. JPEG/PNG renders) instead of linear RAW; they are converted to linear intensity (in float) when loaded.
- `--scaling_factor`: initial scene scale of the shader (default `0.1`, a hand-picked guess for up-to-scale SfM poses). Use `1.0` when the poses are already metric.
- `--light_params`: light/shading parameter file (default `model_parameters.pth` in the working directory).

`points3D.ply` must carry **non-zero normals pointing away from the cameras** (`nx, ny, nz`): the shading uses `relu(n · (point − light))`, whose gradient is zero at `n = 0`, so all-zero normals give a permanently black render and no learning at all (`create_from_pcd` prints a warning in that case). With `--linearize` the evaluation also logs `eval_*/psnr_srgb` (PSNR of the sRGB-encoded images), the number to compare with methods evaluated on the input images.

The Replica `office0` re-render (simulated co-located spot light, ground-truth metric poses, `lit` and uniform-light `even` variants) is packaged for this repo by `scripts/darkgs/build_dataset.py` of the Replica simulation project; each package ships its own `model_parameters.pth` and README with the exact command.

Visualize with SIRB viewer:
```
./SIBR_remoteGaussian_app
```
Then you will be able to steer your light cone by pressing "JKLI" on the keyboard.\
#### Visualize (a checkpoint) after training:
SIBR_gaussianViewer_app is not compatible with this repo. Please try the following:
```
python3 viz_chkpt.py -s data/lab1/ -m output/<xxxxxx-xxx> --start_checkpoint output/<xxxxxx-xxx>/chkpnt30000.pth
```
then in another terminal
```
./SIBR_remoteGaussian_app
```

#### Relighting (I'm working on the release)
Meanwhile, there is no one-true-solution to relighting.\
One quick hack through is when running viz_chkpt.py, uncomment line 129 in scene/lighting.py. And you will also need to brighten, white balance and gamma correct the final render results to make it look good otherwise it is in blue-greenish RAW format.



## Cite
This work is picked up by IROS 2024 as oral presentation!
[Arxiv](https://arxiv.org/abs/2403.10814)
```
@INPROCEEDINGS{zhang2024darkgs,
  author={Tianyi Zhang and Kaining Huang and Weiming Zhi and Matthew Johnson-Roberson},
  booktitle={2024 International Conference on Intelligent Robots and Systems (IROS)}, 
  title={DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark}, 
  year={2024}}
```

## Acknowledgement
* This work is supported by NOAA.
* Copyright 2024 Tianyi Zhang, Carnegie Mellon University. All rights reserved.
