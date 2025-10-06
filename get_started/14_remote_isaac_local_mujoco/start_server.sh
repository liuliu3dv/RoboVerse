#!/bin/bash

# Start Remote Environment Server
# Usage: ./start_server.sh [port] [task]

# Default values
PORT=${1:-8888}
TASK=${2:-"stack_cube"}

echo "Starting Remote Environment Server..."
echo "Port: $PORT"
echo "Task: $TASK"

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

# Start the server
echo "Starting server..."
python example.py server

echo "Server stopped"
