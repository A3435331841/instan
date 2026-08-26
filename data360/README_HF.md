---
license: cc-by-nc-sa-4.0
size_categories:
- n<1K
---

# Introduction
The 360VOTS dataset is a comprehensive benchmark designed for advancing research in omnidirectional visual object tracking (VOT) and video object segmentation (VOS). The dataset comprises 290 sequences in total, with 170 sequences for training and 120 sequences for testing. Each sequence consists of high-resolution 4K RGB frames, capturing a wide variety of real-world scenes and object types from a 360-degree perspective.

For every frame in each sequence, we provide densely annotated segmentation masks, enabling precise evaluation and training for object segmentation and tracking tasks. To facilitate research in VOT, we supply a flexible annotation toolkit that converts the segmentation masks into four distinct types of ground truths commonly used in visual object tracking benchmarks.

Project Page: https://360vots.hkustvgd.com/

Github Repo: https://github.com/HuajianUP/360VOT

# Usage
```python
# Authentication
from huggingface_hub import login
login()

# Download
from datasets import load_dataset
dataset = load_dataset("xuyzshaun/360VOTS")

# List all files
from huggingface_hub import list_repo_files
files = list_repo_files("xuyzshaun/360VOTS", repo_type="dataset")
for file in files: print(file)

# Download specific files
from huggingface_hub import hf_hub_download
file_path = "train/001.zip"
local_file = hf_hub_download(
    repo_id="xuyzshaun/360VOTS",
    filename=file_path,
    repo_type="dataset",
    local_dir="./my_data",  # local destination
    local_dir_use_symlinks=False)
print(f"Downloaded to {local_file}")

```

# Citation
- For only evaluation of omnidirectional visual object tracking (VOT) methods on the 360VOT dataset, please cite the source:
```
@InProceedings{huang360VOT,
author = {Huajian Huang, Yinzhe Xu, Yingshu Chen and Sai-Kit Yeung},
title = {360VOT: A New Benchmark Dataset for Omnidirectional Visual Object Tracking},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2023},
pages = {}
}
```
- For evaluation and training of omnidirectional visual object tracking and segmentation (VOT & VOS) methods on the 360VOTS dataset, please cite the source:
```
@ARTICLE{11090163,
    author={Xu, Yinzhe and Huang, Huajian and Chen, Yingshu and Yeung, Sai-Kit},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    title={360VOTS: Visual Object Tracking and Segmentation in Omnidirectional Videos},
    year={2025},
    volume={47},
    number={11},
    pages={9785-9797},
    doi={10.1109/TPAMI.2025.3591725}
}

@article{xu2025360votsvisualobjecttracking,
    title={360VOTS: Visual Object Tracking and Segmentation in Omnidirectional Videos}, 
    author={Yinzhe Xu and Huajian Huang and Yingshu Chen and Sai-Kit Yeung},
    year={2025},
    eprint={2404.13953},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2404.13953}, 
}
```