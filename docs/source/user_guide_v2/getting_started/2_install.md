
# Installation

## Prepare python environment

Please refer to the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).


Install Isaac Sim
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 4d9914702  # Ensure reproducibility
sudo apt install cmake build-essential
./isaaclab.sh --install
ln -s ~/miniforge3/envs/isaaclab/lib/python3.10/site-packages/isaacsim _isaac_sim  # Optional: for linting
```

Run the Isaac Sim GUI:
```bash
./isaaclab.sh -s
```

Install other dependencies
```bash
conda install opencv numpy==1.26.4  # compatible with pickle files
pip install rootutils loguru rich tyro tqdm
pip install dill  # Optional: load humanoid policy
pip install numpy-quaternion  # Optional: for arnold
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl # Optional: retargeting, multi-embodiment support
```

For developers, please install pre-commit hooks:
```bash
sudo apt install pre-commit
pre-commit install
```

And do install the [black](https://marketplace.cursorapi.com/items?itemName=ms-python.black-formatter) and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) vscode extension.

The `.vscode/settings.json` is configured aligning with the pre-commit hooks. Whenever you save the file, it will be formatted automatically.

## Prepare Data

1. Please download the data from [here](https://drive.google.com/drive/folders/1ORMP3__KIlXettN8eUCF3YQNybZQxzkw) and put it under `./data`.

2. Download the converted isaaclab data from [here](https://drive.google.com/drive/folders/1nF-5SU4nC6S_vgC_NL7E7Rt63Zl9sYqW) and put it under `./data_isaaclab`.

3. Then link `./data/source_data` to `./data_isaaclab/source_data`.
    ```bash
    cd data_isaaclab
    ln -s ../data/source_data source_data
    ```
