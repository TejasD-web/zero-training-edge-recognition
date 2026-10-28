#!/bin/bash
source /mnt/nvme/project/sam2-env/bin/activate
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:/mnt/nvme/project/sam2-env/lib/python3.10/site-packages/nvidia/cu12/lib/:$LD_LIBRARY_PATH
echo "SAM 2.1 env ready"
