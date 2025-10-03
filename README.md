# TryOffDiff

![teaser.gif](references/teaser.gif)

The official repository of the papers:
1. _"TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models"_,
2. _"MGT: Extending Virtual Try-Off to Multi-Garment Scenarios"_

**TL;DR**:
[![arXiv][logo-paper]][paper-arxiv]
[![arXiv][logo-paper2]][paper2-arxiv]
[![Generic badge][logo-hf_spaces]][hf_spaces]
[![Generic badge][logo-hf_models]][hf_models]
[![Generic badge][logo-project_page]][project_page]

---

### 🎉 News
- [2025-09-01]: Demo is accepted at [ICCV'25 Demo Track](https://media.eventhosts.cc/Conferences/ICCV2025/iccv25_main_program.pdf#page=28&zoom=180), will be presented on Oct 21st, 2025.
- [2025-07-25]: Paper1 accepted at [BMVC'25](https://bmvc2025.bmva.org/), will be presented on Nov 24-27, 2025 (tbd).
- [2025-07-11]: Paper2 accepted at [ICCV'25 Workshop](https://cvam-workshop.github.io/), will be presented on Oct 20th, 2025.
- [2025-07-10]: Code for new features made available.
- [2025-04-17]: Paper2 (follow-up work) appeared on arXiv with improvements, _e.g._ multi-garment try-off.
- [2025-03-26]: Demo is accepted at [CVPR'25 Demo Track](https://media.eventhosts.cc/Conferences/CVPR2025/CVPR_main_conf_2025.pdf#page=20&zoom=180), presented on June 13, 2025.
- [2024-12-03]: Training, Inference, and Evaluation scripts made available.
- [2024-11-27]: Paper1 appeared on arXiv.

---

### Usage
Please refer to the [instructions](references/README.md).

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
    ├── config.py        <- Store configuration variables for training and inference
    ├── dataset.py       <- Download and clean datasets VITON-HD & Dress Code
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
@inproceedings{velioglu2025tryoffdiff,
  title     = {TryOffDiff: Virtual-Try-Off via High-Fidelity Garment Reconstruction using Diffusion Models},
  author    = {Velioglu, Riza and Bevandic, Petra and Chan, Robin and Hammer, Barbara},
  booktitle = {BMVC},
  year      = {2025},
  note      = {\url{https://doi.org/nt3n}}
}
@inproceedings{velioglu2025mgt,
  title     = {MGT: Extending Virtual Try-Off to Multi-Garment Scenarios},
  author    = {Velioglu, Riza and Bevandic, Petra and Chan, Robin and Hammer, Barbara},
  booktitle = {ICCVW},
  year      = {2025},
  note      = {\url{https://doi.org/pn67}}
}
```

[project_page]: https://rizavelioglu.github.io/tryoffdiff
[logo-project_page]: https://img.shields.io/badge/Project-Page-purple
[logo-hf_models]: https://img.shields.io/badge/🤗-Models-blue.svg?style=plastic
[logo-hf_spaces]: https://img.shields.io/badge/🤗-Demo-blue.svg?style=plastic
[logo-paper]: https://img.shields.io/badge/arXiv-Paper1-b31b1b.svg?style=plastic
[logo-paper2]: https://img.shields.io/badge/arXiv-Paper2-b31b1b.svg?style=plastic
[hf_datasets]: https://huggingface.co/datasets/rizavelioglu/...
[hf_models]: https://huggingface.co/rizavelioglu/tryoffdiff
[hf_spaces]: https://huggingface.co/spaces/rizavelioglu/tryoffdiff
[paper-arxiv]: https://arxiv.org/abs/2411.18350
[paper2-arxiv]: https://arxiv.org/abs/2504.13078
[cookiecutter_link]: https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter
[cookiecutter]: https://cookiecutter-data-science.drivendata.org/
