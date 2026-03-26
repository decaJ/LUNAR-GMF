#!/bin/bash
# Run script for Steering Unlearn Method
# 
# Usage:
#   bash steering_unlearn/run.sh
#
# Or run directly:
#   cd LUNAR && python steering_unlearn/run_steering_unlearn.py

set -e

# Change to LUNAR directory
cd "$(dirname "$0")/.."

echo "Running Steering Unlearn Method..."
echo "=================================="

# Run the main script
python steering_unlearn/run_steering_unlearn.py

echo ""
echo "Done!"