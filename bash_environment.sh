#!/bin/bash

# Example project environment
# Set up relevant modules
module add miniconda/2.7

# Set up project-specific python environment
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source activate ${scriptdir}/env

# Set Python path to project-specific folder
parentdir="$(dirname "$scriptdir")"
PYTHONPATH="${PYTHONPATH}:${parentdir}"

export project_name="[athena]"
