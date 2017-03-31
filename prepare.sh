#!/bin/sh

echo "Downloading datasets ..." 2>&1
sh download_dataset.sh musk1
sh download_dataset.sh musk2
sh download_dataset.sh elephant
sh download_dataset.sh fox
sh download_dataset.sh tiger
echo "Generating datasets ..." 2>&1
python generate_dataset.py
echo "Cleaning ..." 2>&1
rm musk1.data
rm musk2.data
rm elephant.data
rm fox.data
rm tiger.data
rm -r MilData/
rm MIL-Data-2002-Musk-Corel-Trec9.tgz
