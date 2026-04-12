#!/bin/bash

# Navigate to the project directory
# Adjust this path if necessary or use dirname $0
cd "$(dirname "$0")"

echo "--- Starting Site Update ---"

# 1. Update publications from arXiv using the Python script
echo "Step 1: Fetching latest publications from arXiv..."
python3 update.py

# 2. Build the Jekyll site to reflect changes
# Using the conda environment we set up earlier
echo "Step 2: Rebuilding Jekyll site..."
conda run -n jekyll_env jekyll build

echo "--- Site Update Complete ---"
echo "Refresh your browser to see the changes."
