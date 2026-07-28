# LabGym Command Line Interface (CLI) Guide

This guide details the implementation, command syntax, configuration architecture, and parameter mapping of the headless command-line interface (CLI) for LabGym.

---

## 1. Architecture & GUI Alignment

The LabGym CLI is fully decoupled from the `wxPython` GUI, allowing training, evaluation, and inference workflows to run headlessly on remote cloud servers or High-Performance Computing (HPC) clusters utilizing schedulers like SLURM. 

### GUI-to-CLI Parameter Mapping

The CLI options align directly with parameters defined across the three main GUI panels:

| GUI Panel Field / Option | CLI Option Flag | YAML Configuration Key | Notes / Default Values |
| :--- | :--- | :--- | :--- |
| **Train Categorizers** | | | |
| Select folder of sorted examples | `--data_path` | `data_path` | Dataset directory path |
| Select cache folder for prepared examples | `--out_folder` | `out_folder` | Temporary directory path |
| Specify network complexity/type | `--network_type` | `network_type` | `pattern_recognizer`, `animation_analyzer`, `combnet` |
| Complexity levels (Lv1 to Lv5) | `--level_conv`, `--level_tconv` | `level_conv`, `level_tconv` | Network architectures complexity |
| Input width, height, channels | `--dim_conv`, `--dim_tconv`, `--channel` | `dim_conv`, `dim_tconv`, `channel` | Resolution constraints (e.g., `64`, `32`, `1`) |
| Frames per animation sequence | `--time_step` | `time_step` | Episode window duration (default `15`) |
| Augmentation methods | `--aug_methods` | `aug_methods` | List of strings (e.g. `['rotate', 'flip']`) |
| Save folder for training reports | `--out_path` | `out_path` | Export logs directory |
| On-the-fly training | `--training_onfly` | `training_onfly` | Avoid caching prepared examples (`true`/`false`) |
| **Train Detectors** | | | |
| Select training images folder | `--path_to_trainingimages` | `path_to_trainingimages` | Directory containing raw pictures |
| Select JSON annotation file | `--path_to_annotation` | `path_to_annotation` | COCO-formatted annotations |
| Inferencing framesize | `--inference_size` | `inference_size` | Image frame size (default `480`) |
| Training iterations | `--iteration_num` | `iteration_num` | Backpropagation cycles (default `200`) |
| **Analyze Behaviors (Infer)** | | | |
| Select Categorizer model | `--path_to_categorizer` | `path_to_categorizer` | Trained model path |
| Select videos / images | `--path_to_videos` | `path_to_videos` | Input directory or list of file paths |
| Select output folder | `--result_path` | `result_path` | Target output directory |
| Specify object detector method | `--use_detector`, `--path_to_detector` | `use_detector`, `path_to_detector` | Toggle and path for Det2 model |
| Analysis start offset | `--t` | `t` | Starting second offset (default `0.0`) |
| Analysis duration | `--duration` | `duration` | Total evaluation length (default `0.0` = end) |
| Number of animals | `--animal_number` | `animal_number` | Total target counts (integer or JSON map) |
| Target behaviors to annotate | `--behavior_to_include` | `behavior_to_include` | List of target behavior names (default `['all']`) |
| Target measurements | `--parameter_to_analyze` | `parameter_to_analyze` | Kinematic options |

---

## 2. CLI Command Syntax & Option Overrides

Every parameter mapped in the configuration files is exposed as an explicit CLI option flag. In-line flags dynamically override matching keys passed inside a YAML file.

### Default GUI Fallback
If you run `labgym` without any subcommands, it will fall back to starting the desktop GUI interface:
```bash
labgym
```

### Subcommand `train`
Headless training execution for Categorizers (sub-networks) or Detectors (object detection).

*   **Syntax**:
    ```bash
    labgym train --config <file.yaml> [inline-overrides]
    ```

*   **Inline Parameters**:
    *   `--network_type`: Choice of `pattern_recognizer`, `animation_analyzer`, `combnet`, `detector`
    *   `--model_path`: Output model folder target
    *   `--data_path`: Training examples directory path
    *   `--training_onfly`: Enable/disable on-the-fly training (accepts `true` or `false`)
    *   `--path_to_annotation`: Path to COCO annotation file (required if `--network_type detector`)
    *   `--path_to_trainingimages`: Path to training images folder (required if `--network_type detector`)
    *   `--iteration_num`: Detector training iterations (integer)
    *   `--inference_size`: Detector inference frame size (integer)

### Subcommand `test`
Headless evaluation of a model (Categorizer or Detector) against a testing dataset. 

For **Categorizer** evaluation, it automatically calculates mathematically exact centroids of behaviors in the high-dimensional penultimate-layer feature space using **Cosine Similarity**, executing the full diagnostics pipeline and rendering a formatted UMich-style `Triage_Action_Plan.pdf` inside the output `LabGym_Diagnostics/` folder.

For **Detector** evaluation, it runs headless bounding box validation against annotated images and calculates mean Average Precision (mAP).

*   **Syntax**:
    ```bash
    labgym test --config <file.yaml> [inline-overrides]
    ```

*   **Inline Parameters**:
    *   `--model_type`: Choice of `categorizer` or `detector` (default: `categorizer`)
    *   `--path_to_annotation`: Path to COCO JSON annotation file (required if `--model_type detector`)
    *   `--groundtruth_path`: Directory containing true behavior classes (for Categorizers) or testing images directory (for Detectors)
    *   `--model_path`: Trained model folder path under evaluation
    *   `--result_path`: Output statistics/diagnostics folder path
    *   `--redundant`: [Categorizer only] List of space-separated true-to-pred error pairs to triage as Redundant (Hypothesis 1)
    *   `--emergent`: [Categorizer only] List of error pairs to triage as Emergent Behaviors (Hypothesis 2)
    *   `--generalization`: [Categorizer only] List of error pairs to triage as Poor Generalization (Hypothesis 3)

### Subcommand `infer`
Headless tracking and behavior categorization on raw input videos or images.

*   **Syntax**:
    ```bash
    labgym infer --config <file.yaml> [inline-overrides]
    ```

*   **Inline Parameters**:
    *   `--path_to_videos`: Space-separated video files or a folder containing raw files
    *   `--result_path`: Target results folder
    *   `--path_to_categorizer`: Trained model path
    *   `--use_detector`: Toggle detector usage (`true`/`false`)
    *   `--path_to_detector`: Object detection model path
    *   `--t`: Duration offset start (seconds)
    *   `--duration`: Execution frame window (seconds)

---

## 3. YAML Configuration Schema Examples

Below are standard baseline configuration file structures matching each workflow subcommand.

### `train_config.yaml` (Categorizer Sub-Network Training)
```yaml
# Categorizer sub-network training
network_type: "combnet"
data_path: "./prepared_dataset"
model_path: "./models/my_new_categorizer"
out_folder: "./cached_frames"

# Architecture hyperparameters
dim_conv: 64
dim_tconv: 32
channel: 1
time_step: 15
level_conv: 2
level_tconv: 1

# Training parameters
training_onfly: false
augvalid: true
include_bodyparts: true
std: 0.0
background_free: true
black_background: true
behavior_mode: 0
social_distance: 0.0
```

### `train_detector_config.yaml` (Detector Model Training)
```yaml
# Detector training
network_type: "detector"
model_path: "./models/my_new_detector"

# Dataset paths
path_to_annotation: "./dataset/annotations.json"
path_to_trainingimages: "./dataset/images"

# Hyperparameters
iteration_num: 1000
inference_size: 480
```

### `test_config.yaml` (Diagnostics & Triage Plan Generation)
```yaml
# Headless testing parameters
groundtruth_path: "./ground_truth_examples"
model_path: "./models/my_new_categorizer"
result_path: "./evaluation_results"

# Triage Hypothesis mapping (Optional)
# If left blank, all systematic major confusions automatically calculate
# centroids and embed their reference captures directly.
redundant:
  - "behavior_jump -> behavior_leap"
emergent:
  - "behavior_groom -> behavior_scratch"
generalization:
  - "behavior_walk -> behavior_run"
```

### `infer_config.yaml` (Behavior Tracking & Prediction)
```yaml
# Batch inference setup
path_to_videos:
  - "./raw_videos/video_batch_1"
result_path: "./output_results"
path_to_categorizer: "./models/my_new_categorizer"

# Object Detection configs
use_detector: true
path_to_detector: "./models/my_new_detector"
detector_batch: 4
animal_kinds:
  - "mouse"

# Video properties decoding & constraints
decode_animalnumber: true
stable_illumination: true
t: 0.0
duration: 60.0
```

---

## 4. SLURM SBatch Execution Example

Below is a baseline script configuration for submitting parallel parameters sweeping runs on SLURM clusters.

```bash
#!/bin/bash
#SBATCH --job-name=labgym_sweep
#SBATCH --output=logs/labgym_%A_%a.out
#SBATCH --error=logs/labgym_%A_%a.err
#SBATCH --array=1-3
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load modules / Activate virtual env
module load python/3.10
source venv/bin/activate

# Define configurations list for array jobs
declare -A RATES
RATES[1]=0.001
RATES[2]=0.0005
RATES[3]=0.0001

LR=${RATES[$SLURM_ARRAY_TASK_ID]}

# Execute using baseline config and inline parameter overrides
labgym train \
  --config baseline_train_config.yaml \
  --model_path "./models/model_sweep_${SLURM_ARRAY_TASK_ID}" \
  --learning_rate $LR
```
