# Projects Overview

Guide to directory structure, configuration setups, and experimental replication across `mmdetection3d/projects`.

---

## Directory Structure

* **`/subsets_creation`**  
  * Generate all dataset splits for nuScenes.  
  * Produce required `.pkl` metadata files for training.

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

* **FedDyn, FedMC, FedSelect:**
* Use a 10-round FedAvg warm-up phase to establish a stable initialization (all weights except the image backbone begin randomly initialized). This step significantly stabilizes optimization and boosts final metrics.


* **FedCKA:**
* Do **not** use the FedAvg warm-up phase; pre-aggregating weights degrades representation alignment and harms downstream performance.
