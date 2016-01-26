#!/bin/bash

# Example project environment
# Set up relevant modules
module add miniconda/2.7

# Set up project-specific python environment
source activate ~/Athena/env

# Set Python path to project-specific folder
PYTHONPATH="${PYTHONPATH}:~/Athena/"

export project_name="[athena]"
