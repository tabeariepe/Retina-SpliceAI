#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=initialize
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI/slurm/initialize.out

cd /home/scripts/
pwd
 
for i in {1..5}; 
    do python3 train_model.py $i retina standard initialize;
done
