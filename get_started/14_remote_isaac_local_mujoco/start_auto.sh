#!/bin/bash

# Auto-start remote server and connect client
# Usage: ./start_auto.sh [remote_host] [task] [num_envs]

# Default values
REMOTE_HOST=${1:-"user@remote_server"}
TASK=${2:-"stack_cube"}
NUM_ENVS=${3:-1}

echo "Auto-starting remote environment..."
echo "Remote host: $REMOTE_HOST"
echo "Task: $TASK"
echo "Number of environments: $NUM_ENVS"

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roboverse

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "roboverse" ]]; then
    echo "Error: Failed to activate roboverse conda environment"
    echo "Please make sure the environment exists and try again"
    exit 1
fi

echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Change to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start client with auto-start remote server
echo "Starting client with auto-start remote server..."
python example.py \
    --mode client \
    --remote_host "$REMOTE_HOST" \
    --auto_start_remote \
    --task "$TASK" \
    --num_envs "$NUM_ENVS"

echo "Client stopped"
