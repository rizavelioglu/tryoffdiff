# TryOffDiff

![teaser.gif](references/teaser.gif)

The official repository of the paper: _"TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models"_.

**TL;DR**:
[![arXiv][logo-paper]][paper-arxiv]
[![Generic badge][logo-hf_spaces]][hf_spaces]
[![Generic badge][logo-hf_models]][hf_models]
[![Generic badge][logo-project_page]][project_page]

---

### TODOs
- [x] Training script
- [x] Inference script
- [x] Eval script
- [ ] Ablation models' scripts
- [ ] Baseline models' scripts

---


### Install
Create a new Conda environment:
```bash
conda create -n vtoff python=3.11
conda activate vtoff
```

Then, clone the repository, install the required packages:
```bash
git clone https://github.com/rizavelioglu/tryoffdiff.git
cd tryoffdiff
pip install -e .
```

### Dataset
Download the original VITON-HD dataset and extract it to `"./data/vitonhd"`:
```bash
python tryoffdiff/dataset.py download-vitonhd  # For a different location: output-dir="<other-folder>"
```
As mentioned in the paper, the original dataset contains duplicates, and some training samples are leaked into the test
set. Clean these with the following command:
```bash
python tryoffdiff/dataset.py clean-vitonhd  # Default: `data-dir="./data/vitonhd"`
```

### Training
For faster training, pre-extract the image features and save them instead of extracting them during training.

#### Step 1: Encode garment images with VAE
```bash
python tryoffdiff/dataset.py vae-encode-vitonhd \
 --data-dir "./data/vitonhd/" \
 --model-name "sd14" \
 --batch-size 16
 ```

#### Step 2: Encode model(conditioning) images with SigLIP
```bash
python tryoffdiff/dataset.py siglip-encode-vitonhd \
 --data-dir "./data/vitonhd/" \
 --batch-size 64
 ```

#### Step 3: Train `TryOffDiff`
```bash
accelerate launch --multi_gpu --num_processes=4 tryoffdiff/modeling/train.py tryoffdiff \
 --save-dir "./models/" \
 --data-dir "./data/vitonhd-enc-sd14/" \
 --model-class-name "TryOffDiff" \
 --mixed-precision "no" \
 --learning-rate 0.0001 \
 --train-batch-size 16 \
 --num-epochs 1201 \
 --save-model-epochs 100 \
 --checkpoint-every-n-epochs 100
```

> **Note**: See [config.py(TrainingConfig)](tryoffdiff/config.py) for all possible arguments, e.g. set `resume_from_checkpoint` to resume
training from a specific checkpoint.

#### Ablations
Other models presented in the ablation study can be trained similarly. View all available models:
```bash
python tryoffdiff/modeling/train.py --help
```

\[...Work in progress...\]

[//]: # (> Example: Train the `LDM-1` model:)
[//]: # (> ```bash)
[//]: # (> python tryoffdiff/modeling/train.py ldm1 \)
[//]: # (>)
[//]: # (>)
[//]: # (>)
[//]: # (>)
[//]: # (>)
[//]: # (> ```)

### Inference
Each model has its own command. View all available options:
```bash
python tryoffdiff/modeling/predict.py --help
```

> Example: Run inference with `TryOffDiff`:
> ```bash
> python tryoffdiff/modeling/predict.py tryoffdiff \
>  --model-dir "/model_20241007_154516/" \
>  --model-filename "model_epoch_1200.pth" \
>  --batch-size 8 \
>  --num-inference-steps 50 \
>  --seed 42 \
>  --guidance-scale 2.0
> ```
which saves predictions to `"<model-dir>/preds/"` as `.png` files.

> **Note**: See [config.py(InferenceConfig)](tryoffdiff/config.py) for all possible arguments,
e.g. use the `--all` flag to run inference on the entire test set.

> **Note**: The paper uses the PNDM noise scheduler. For HuggingFace Spaces we use the EulerDiscrete scheduler.

### Evaluation

Evaluate the predictions using:
```bash
python tryoffdiff/modeling/eval.py \
 --gt-dir "./data/vitonhd/test/cloth/" \
 --pred-dir "<prediction-dir>" \
 --batch-size 32 \
 --num-workers 4
 ```
which prints the results to the console.
Specifically, we use the following libraries for the implementations of the metrics presented in the paper:
- `pyiqa`: `SSIM`, `MS-SSIM`, `CW-SSIM`, and `LPIPS`,
- `clean-fid`: `FID`, `CLIP-FID`, and `KID`,
- `DISTS-pytorch`: `DISTS`

In addition, we offer a simple GUI for visualizing predictions alongside their evaluation metrics. This tool displays the ground truth and predicted images side-by-side while providing metrics for the entire test set:
```bash
python tryoffdiff/modeling/eval_vis.py \
 --gt-dir "./data/vitonhd/test/cloth/" \
 --pred-dir "<prediction-dir>"
```

---

### Project Organization
The following project/directory structure is adopted: [Cookiecutter Data Science-v2 by DrivenData][cookiecutter]
![cookiecutter-symbol][cookiecutter_link]

```
├── notebooks/           <- Jupyter notebooks
├── references/          <- Manuals and all other explanatory materials.
├── LICENSE
├── README.md
├── pyproject.toml       <- Project configuration file with package metadata
|
└── tryoffdiff/          <- Source code for use in this project.
    ├── modeling/
    │   ├── __init__.py
    │   ├── eval.py      <- Code to evaluate models
    │   ├── model.py     <- Model implementations
    │   ├── predict.py   <- Code to run model inference with trained models
    │   └── train.py     <- Code to train models
    |
    ├── __init__.py      <- Makes `tryoffdiff` a Python module
    ├── config.py        <- Store configuration variables
    ├── dataset.py       <- Download and clean VITON-HD dataset
    ├── features.py      <- Code to create features for modeling
    └── plots.py         <- Code to create visualizations
```

### Acknowledgements
Our code relies on PyTorch, with [🤗 Diffusers](https://github.com/huggingface/diffusers) for diffusion model components
and [🤗 Accelerate](https://github.com/huggingface/accelerate) for multi-GPU training.\
We adopt [Stable Diffusion-v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) as the base model and use
[SigLIP](https://huggingface.co/google/siglip-base-patch16-512) as the image encoder.\
For evaluation, we use [IQA_PyTorch](https://github.com/chaofengc/IQA-PyTorch),
[clean-fid](https://github.com/GaParmar/clean-fid),
and [DISTS-pytorch](https://github.com/dingkeyan93/DISTS).


### License
**_TL;DR_**: Not available for commercial use, unless the FULL source code is open-sourced!\
This project is intended solely for academic research. No commercial benefits are derived from it.\
The code, datasets, and models are published under the [Server Side Public License (SSPL)](LICENSE).


### Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation:
```
@article{velioglu2024tryoffdiff,
  title     = {TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models},
  author    = {Velioglu, Riza and Bevandic, Petra and Chan, Robin and Hammer, Barbara},
  journal   = {arXiv},
  year      = {2024},
  note      = {\url{https://doi.org/nt3n}}
}
```

[project_page]: https://rizavelioglu.github.io/tryoffdiff
[logo-project_page]: https://img.shields.io/badge/Project-Page-purple
[logo-hf_models]: https://img.shields.io/badge/🤗-Models-blue.svg?style=plastic
[logo-hf_spaces]: https://img.shields.io/badge/🤗-Demo-blue.svg?style=plastic
[logo-paper]: https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?style=plastic
[hf_datasets]: https://huggingface.co/datasets/rizavelioglu/...
[hf_models]: https://huggingface.co/rizavelioglu/tryoffdiff
[hf_spaces]: https://huggingface.co/spaces/rizavelioglu/tryoffdiff
[paper-arxiv]: https://arxiv.org/abs/2411.18350
[cookiecutter_link]: https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter
[cookiecutter]: https://cookiecutter-data-science.drivendata.org/
