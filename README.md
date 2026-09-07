# FedCKA: Representation-Guided Layer Personalization for Federated 3D Perception Across Driving Domains

Anonymous code repository accompanying our paper submission.

## Overview

This repository contains the code, configurations, and evaluation scripts for **FedCKA**, a personalized federated learning method for 3D object detection under heterogeneous driving domains.

FedCKA uses **Centered Kernel Alignment (CKA)** to measure layer-wise representation similarity between locally trained client models and the global model. Based on these similarities, client-specific aggregation masks determine which layers remain globally shared and which are locally personalized.

Our experiments evaluate federated learning across domains defined by differences in:

- Location
- Time of day
- Weather

using the **nuScenes** dataset and the **Cross-Modal Transformer (CMT)** 3D object detector.

## Repository Structure

```text
.
├── analysis/           # Evaluation and result-processing scripts
├── configs/            # Experiment configurations
├── federated/          # Federated aggregation and personalization methods
├── tools/              # Training and evaluation utilities
└── README.md
```

The exact directory structure may differ slightly depending on the experiment.

## Methods

The repository contains implementations or experiment configurations for the federated methods evaluated in the paper, including:

* FedAvg
* FedDyn
* PCGrad
* FedBN
* FedRep
* FedSelect
* FedMC
* **FedCKA (ours)**

FedCKA supports two CKA-based personalization strategies:

* **Percentage-based masking:** personalizes the lowest-similarity layers until a predefined parameter ratio is reached.
* **Threshold-based masking:** dynamically personalizes layers whose CKA similarity falls below a predefined threshold. Layers may become globally shared again in later rounds if their representations realign.

## Dataset

Experiments use the **nuScenes** train/validation dataset.

The data are divided into five client domains based on combinations of location, illumination, and weather:

| Client | Location  | Time  | Weather |
| ------ | --------- | ----- | ------- |
| A      | Boston    | Day   | Clear   |
| B      | Boston    | Day   | Rain    |
| C      | Singapore | Day   | Clear   |
| D      | Singapore | Night | Clear   |
| E      | Singapore | Night | Rain    |

The nuScenes dataset itself is **not included** in this repository and must be downloaded separately from the official source.

## Setup

The experiments are based on **MMDetection3D** and the **CMT** detector.

Please install the required dependencies before running the experiments. Environment and configuration files included in this repository provide the versions used for the paper experiments.

## Running Experiments

Training follows a federated procedure in which each client trains locally before client models are processed by the selected federated aggregation method.

Experiment configurations and scripts for reproducing the reported results are provided in the corresponding directories.

The main FedCKA experiments use:

* 5 federated clients
* 40 communication rounds
* 1 local epoch per round
* CKA-based masking at each communication round
* 10 local samples for estimating layer-wise representation similarity

## Evaluation

The main evaluation metrics are:

* **mAP** — mean Average Precision
* **NDS** — nuScenes Detection Score
* **CAP** — Car Average Precision

Additional scripts are provided for cross-domain evaluation, aggregation of repeated runs, and analysis of personalization masks.

## Anonymity

This repository has been anonymized for peer review. Author identities, institutional information, and other identifying metadata have been removed.

## Citation

Citation information will be added following the review process.

## License

The repository builds upon existing open-source projects, including MMDetection3D and CMT. Please refer to their respective licenses for code originating from those projects.

```
```
