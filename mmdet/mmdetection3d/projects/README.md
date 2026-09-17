# Projects Overview

Guide to directory structure, configuration setups, and experimental replication across `mmdetection3d/projects`.

---

## Directory Structure

* **`/subsets_creation`**  
  * Generate all dataset splits for nuScenes.  
  * Produce required `.pkl` metadata files for training.

* **`/cmt`**  
  * Contains the custom training hooks for CMT.  
  * Includes client-side behavior hooks such as FedCKA, FedDyn, FedSelect, and other custom aggregation utilities.

* **`/cmt_40_epoch`**  
  * `experiments_config.sh`: Defines core environment variables required by all `.sbatch` scripts.  
  * `tools/modular_merging.py`: Implementation of all model merging strategies.  
  * Subdirectories: Each experiment contains its own dedicated directory with at least one runnable script.

* **`/cmt_full`**  
  * Complete reproduction of the original CMT baseline configuration and benchmarks.

* **`/configs`**  
  * Contains all configuration files used for the experiments.

* **`/analysis`**  
  * `/class_balance`: Reproduce the nuScenes class distribution overview.  
  * `/qualitative_results`: Generate qualitative prediction visualizations reported in the paper.

* **`/mmdet3d_plugin`**  
  * Custom extensions built on the Cross-Modal Transformer codebase.  
  * Adds FP16 support adapted and metrics adapted for small batch sizes under low-sample domain regimes.

---

## Environment & Path Configuration

Before launching jobs, replace placeholder strings across all configuration files:

* `YOUR_WORKDIR_PATH_HERE`
* `YOUR_HOME_DIR`
* `YOUR_NAME_HERE`
* `YOUR_PATH_HERE`
* `YOUR_NUSCENES_PATH_HERE`
* `YOUR_TMPDIR_PATH_HERE`

For batch replacement (recommended for all except `YOUR_PATH_HERE`):

```bash
cd /mmdet/mmdetection3d/projects
find . -type f -exec grep -lF "YOUR_WORKDIR_PATH_HERE" {} + | xargs -r sed -i 's|YOUR_WORKDIR_PATH_HERE|/your/actual/workdir/path/here|g'

```

---

## Aggregation & Warm-Up Notes

* **FedDyn, FedMC, FedSelect, FedCKA rho setting:**
* Use a 10-round FedAvg warm-up phase to establish a stable initialization (all weights except the image backbone begin randomly initialized). This step significantly stabilizes optimization and boosts final metrics.

* **FedCKA tau setting:**
* Do **not** use the FedAvg warm-up phase; for layers are allowed to flow back if they align representationally. Using the 10 round warm-up deteriorates performance here, while for above descriped methods performance increases. It does use the first round FedAvg logic to initialize the same model for all clients.

* **FedCKA LocalSim setting:**
* Including LocalSim (Federated learning with partial model personalization, Pillutla et al., ICML 2022) for personalized subsets stabilizes and improves training. Replacing the config file and including the global model path is required for this. See FedSelect implementatio for details.


---

## MMDetection3D runner settings

The experiment scripts support both MMDetection3D runner modes:

* `--slurm` submits distributed training through Slurm. Use this mode when you have access to a Slurm allocation and the required account, partition, and resource permissions.
* `--pytorch` launches distributed training directly with PyTorch. Use this mode when Slurm submission is unavailable or when running on a local machine or an already allocated node.

Choose exactly one mode for each run. The two modes should use equivalent GPU, CPU, memory, and time-limit settings so that results remain comparable. Check the selected `.sbatch` script and adapt its Slurm options, environment paths, and launcher arguments to your cluster before submitting the job.

