## Install

```bash
cd roboverse_learn/utils/diffusion_policy

pip install -e .

cd ../../../

pip install pandas wandb
```

Register for a Weights & Biases (wandb) account to obtain an API key.

## Collect and process data

```bash
./roboverse_learn/collect_demo.sh
```

## Train and eval

```bash
./roboverse_learn/run.sh
```
