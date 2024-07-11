#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI-2/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=pacbio
#SBATCH --exclude=dlc-electabuzz
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI-2/slurm/pacbio.out

pip3 install pysam
pip3 install pyfaidx

cd /home/validation_scripts/

python3 predictions_for_pacbio_junctions.py > pacbio_junctions.txt
