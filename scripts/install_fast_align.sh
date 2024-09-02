#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define the root directory of the repository
REPO_ROOT="$SCRIPT_DIR/.."

# Create tools directory if it doesn't exist
if [ ! -d "$REPO_ROOT/tools" ]; then
  mkdir -p "$REPO_ROOT/tools"
  cd "$REPO_ROOT/tools"

  # Clone and build fast_align
  git clone https://github.com/clab/fast_align.git
  cd fast_align
  mkdir build
  cd build
  cmake ..
  make
else
  echo "Tools directory already exists, skipping installation."
fi