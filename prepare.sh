#!/bin/sh

echo "Downloading datasets ..." 2>&1
sh download_dataset.sh musk1
sh download_dataset.sh musk2
sh download_dataset.sh elephant
sh download_dataset.sh fox
sh download_dataset.sh tiger
echo "Generating datasets ..." 2>&1
python generate_dataset.py
