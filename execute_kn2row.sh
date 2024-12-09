#!/bin/bash
#SBATCH --account=sxk1942_csds451     # Account name
#SBATCH --partition=markov_gpu        # Partition name
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --constraint=gpul40s          # GPU constraint
#SBATCH --job-name=ai3_setup          # Job name
#SBATCH --output=ai3_setup.out        # Output file
#SBATCH --error=ai3_setup.err         # Error file
#SBATCH --time=01:00:00               # Maximum runtime (1 hour)
#SBATCH --pty                         # Interactive session

# Load required modules
module load Python/3.11.5-GCCcore-13.2.0
module use /home/nkt8/.usr/local/share/modulefiles
module load SYCL/2024.0.1.46

# Install dependencies
pip install "torch>=2.4"
pip install torchvision

# Change directory to ai3 and install the package
pip install .

python test_cnn.py

/bin/bash