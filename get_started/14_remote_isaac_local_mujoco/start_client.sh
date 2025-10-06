#!/bin/bash

# Start Remote Environment Client
# Usage: ./start_client.sh [server_host] [server_port]

# Default values
SERVER_HOST=${1:-"localhost"}
SERVER_PORT=${2:-8888}

echo "Starting Remote Environment Client..."
echo "Server: $SERVER_HOST:$SERVER_PORT"

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

# Start the client
echo "Starting client..."
python example.py

echo "Client stopped"
