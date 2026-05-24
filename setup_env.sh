#!/bin/bash
# Runway Shared Manifold Domain Transfer.
# Source this file to set up the Python path for development
# Usage: source setup_env.sh

# Get the directory where this script is located (repo root)
export RUNWAY_SMDT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add src directory to Python path
export PYTHONPATH="${RUNWAY_SMDT_ROOT}/src:${PYTHONPATH}"

# Activate the virtual environment if it exists
if [ -d "${RUNWAY_SMDT_ROOT}/.runway_domain_transfer_env" ]; then
    source "${RUNWAY_SMDT_ROOT}/.runway_domain_transfer_env/bin/activate"
    echo "Activated .runway_domain_transfer_env virtual environment"
else
    echo "Virtual environment .runway_domain_transfer_env not found"
fi

echo "Added ${RUNWAY_SMDT_ROOT}/src to PYTHONPATH"
echo "Environment setup complete"