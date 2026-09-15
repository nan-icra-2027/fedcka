# FedCKA: Representation-Guided Layer Personalization for Federated 3D Perception

Anonymous code repository accompanying our paper submission.

## Overview

**FedCKA** is a personalized federated learning method for 3D object detection under heterogeneous driving domains. It uses Centered Kernel Alignment (CKA) to measure layer-wise representation similarity, creating dynamic client-specific aggregation masks that determine which layers remain globally shared and which are locally personalized.

## Repository Structure

```text
.
├── mmdet/                              # Main project directory
│   ├── image/                          # Apptainer definition file and container setup
│   │   └── README.md                   # Container build and verification instructions
│   └── mmdetection3d/                  # Core framework and project workspace
│       ├── mmdet3d/                    # Customized mmdet3d runner
│       ├── projects/                   # All experiments and configurations
│       │   ├── analysis/               # Evaluation, class balance, and qualitative results
│       │   ├── cmt_40_epoch/           # Main federated experiments and model merging tools
│       │   ├── cmt_full/               # Full CMT baseline replication
│       │   ├── mmdet3d_plugin/         # Custom FP16 CMT plugin for small batch sizes
│       │   ├── subsets_creation/       # nuScenes split generation and .pkl metadata files
│       │   └── README.md               # Projects guide, path configurations, and setups
│       ├── tools/                      # Adjusted train and evaluation scripts
│       └── README.md                   # Base MMDetection3D clone and checkpoint setup
└── README.md                           # Main project overview, NDS results, and citations

```

## Results (NDS)

Comparison of Federated Learning methods across nuScenes domains. Performance is reported as **nuScenes Detection Score (NDS)**.

| Method | Domain A | Domain B | Domain C | Domain D | Domain E | Avg. |
| --- | --- | --- | --- | --- | --- | --- |
| *Centralized* | *0.66* | *0.68* | *0.67* | *0.64* | *0.58* | *0.66* |
| *Own Domain Only* | *0.64* | *0.51* | *0.62* | *0.24* | *0.07* | *0.57* |
| FedAvg [1] | 0.37 | 0.36 | 0.38 | 0.35 | 0.33 | 0.37 |
| FedDyn [2] | 0.48 | 0.46 | 0.48 | 0.43 | 0.35 | 0.47 |
| PCGrad [3, 4] | 0.41 | 0.41 | 0.44 | 0.39 | 0.34 | 0.42 |
| FedRep [5] | 0.43 | 0.40 | 0.44 | 0.41 | 0.33 | 0.43 |
| FedBN [6] | 0.41 | 0.41 | 0.43 | 0.36 | 0.33 | 0.41 |
| FedMC [7] | 0.56 | 0.56 | 0.58 | 0.47 | 0.32 | 0.56 |
| FedSelect [8] | 0.59 | 0.51 | 0.56 | 0.40 | 0.34 | 0.55 |
| **FedCKA (Ours)** | **0.65** | **0.63** | **0.64** | **0.59** | **0.50** | **0.63** |


## Download Dependencies & Checkpoints (Hugging Face)

The required `.sif` container, Flash Attention wheel, and pretrained checkpoints are hosted anonymously on Hugging Face. 

### 1. Container & Wheel Files
Download the image components directly into the `/image` directory:

```bash
cd /YOUR_PATH_HERE/mmdet/image
wget -c [https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/mmdet3d_v1rc5.sif](https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/mmdet3d_v1rc5.sif)
wget -c [https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl](https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl)

```

### 2. Pretrained Checkpoints

Create the checkpoints directory and download the required `.pth` weights into it:

```bash
mkdir -p /YOUR_PATH_HERE/mmdet/mmdetection3d/ckpts
cd /YOUR_PATH_HERE/mmdet/mmdetection3d/ckpts
wget -c [https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/nuim_r50.pth](https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/nuim_r50.pth)
wget -c [https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/fcos3d_vovnet_imgbackbone-remapped.pth](https://huggingface.co/datasets/Anon-fedcka-ICRA/essentials/resolve/main/fcos3d_vovnet_imgbackbone-remapped.pth)

```


## Anonymity & Citation

This repository is anonymized for peer review. Author identities and institutional information have been removed. 

```bibtex
@inproceedings{anonymous2026fedcka,
  title={FedCKA: Representation-Guided Layer Personalization for Federated 3D Perception Across Driving Domains},
  author={Anonymous},
  booktitle={Under Review},
  year={2026}
}

```

## AI Usage Disclosure

During the development of this codebase, the authors utilized ChatGPT, Google Gemini, and GitHub Copilot in Agent Mode as programming assistants. The system was used strictly as an assistive tool for debugging, expanding existing functions, and generating boilerplate code. All AI-generated code was thoroughly reviewed, tested, and modified by the authors to ensure correctness and compatibility within the MMDetection3D environment. The authors assume full responsibility for the functionality, logic, and integrity of the code in this repository.

### References

* [1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, "Communication efficient learning of deep networks from decentralized data," in *Proc. AISTATS*, 2017, pp. 1273–1282.
* [2] D. A. E. Acar, Y. Zhao, R. Matas Navarro, M. Mattina, P. N. Whatmough, and V. Saligrama, "Federated learning based on dynamic regularization," in *Proc. ICLR*, 2021.
* [3] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, "Gradient surgery for multi-task learning," in *Proc. NeurIPS*, 2020, pp. 5824–5836; and X. Zhang, W. Sun, and Y. Chen, "Tackling the non-IID issue in heterogeneous federated learning by gradient harmonization," *IEEE Signal Processing Letters*, vol. 31, pp. 2595–2599, 2024.
* [4] L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, "Exploiting shared representations for personalized federated learning," in *Proc. ICML*, 2021, pp. 2089–2099.
* [5] X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, "FedBN: Federated learning on non-IID features via local batch normalization," in *Proc. ICLR*, 2021.
* [6] Y. Gao, X. He, and Y. Chen, "Personalized federated learning algorithm based on information content model customization," in *Proc. CAMMIC*, 2025, pp. 834–838.
* [7] R. Tamirisa, C. Xie, W. Bao, A. Zhou, R. Arel, and A. Shamsian, "FedSelect: Personalized federated learning with customized selection of parameters for fine-tuning," in *Proc. CVPR*, 2024, pp. 29485–29494.
* [8] MMDetection3D Contributors, "MMDetection3D: OpenMMLab next-generation platform for general 3D object detection," 2020. [Online]. Available: https://github.com/open-mmlab/mmdetection3d
* [9] J. Yan et al., "Cross modal transformer: Towards fast and robust 3D object detection," in *Proc. ICCV*, 2023, pp. 18268–18278.

## License

This repository builds upon MMDetection3D [8] and CMT [9]. Please refer to their respective open-source licenses for inherited code.
