# RoboVerse VLA Training Pipeline

A brief workflow for training Vision-Language-Action (VLA) models using RoboVerse robotic manipulation data and OpenVLA framework.



## Workflow

### Step 1: Collect Demonstration Trajectories

Generate robotic demonstration data using the RoboVerse simulation environment:

```bash
python scripts/advanced/collect_demo.py \
  --sim=mujoco --task=pick_butter --headless --run_all
```

**Parameters:**
- `--sim`: Simulation backend (mujoco)
- `--task`: Specific manipulation task (pick_butter)
- `--headless`: Run without GUI for efficiency
- `--run_all`: Execute all available episodes

### Step 2: Data Format Conversion to RLDS

Convert collected demonstrations to the Robot Learning Dataset Specification (RLDS) format:

```bash
# Navigate to RLDS utilities directory
cd roboverse_learn/vla/rlds_utils/roboverse/

# Create symbolic link for demo data
# Note: Adjust folder name according to your task & simulation setup
mkdir -p demo
ln -s /absolute/path/to/roboverse_demo/demo_mujoco/pick_butter- demo/pick_butter-



# Set up conversion environment
conda env create -f ../environment_ubuntu.yml
conda activate rlds_env

# Convert to RLDS format
tfds build --overwrite
```

**Output:** Converted dataset stored in `~/tensorflow_datasets/roboverse_dataset/`

### Step 3: OpenVLA Model Fine-tuning

#### 3.1 Environment Setup

```bash
# Navigate to third-party dependencies
cd thirdparty

# Clone OpenVLA repository
git clone https://github.com/openvla/openvla.git
```

#### 3.2 Installation

Follow the official OpenVLA installation guide:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch with CUDA support
# Check https://pytorch.org/get-started/locally/ for platform-specific instructions
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install OpenVLA package
cd openvla
pip install -e .

# Install Flash Attention 2 for efficient training
# Reference: https://github.com/Dao-AILab/flash-attention
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja installation (should return exit code "0")
pip install "flash-attn==2.5.5" --no-build-isolation
```

#### 3.3 Fine-tuning Execution

```bash
cd roboverse_learn/vla/
bash finetune_roboverse.sh
```

**Important:** Ensure dataset and model paths in the script match your actual directory structure.

### Step 4: Model Evaluation

Assess the trained model's performance on specific tasks:

```bash
python roboverse_learn/vla/vla_eval.py \
  --model_path openvla_runs/path/to/checkpoint \
  --task pick_butter
```

## Data Pipeline

```
Demo Collection → RLDS Conversion → VLA Fine-tuning → Model Evaluation
     ↓                    ↓                ↓                ↓
Raw Trajectories → Standardized Format → Trained Model → Performance Metrics
```

## Dataset Information

- **Format:** RLDS (Robot Learning Dataset Specification)
- **Storage Location:** `~/tensorflow_datasets/roboverse_dataset/`

## Configuration Notes

- Adjust simulation parameters in `collect_demo.py` based on your specific tasks
- Modify dataset paths in `finetune_roboverse.sh` to match your environment

