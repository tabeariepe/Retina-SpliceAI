#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=standard
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI/slurm/standard.out

#cd /home/scripts/

for i in {4..5}; 
    do python3 train_model.py $i retina standard train > ../output_train/SpliceAI_standard_retina${i}.txt;
done
