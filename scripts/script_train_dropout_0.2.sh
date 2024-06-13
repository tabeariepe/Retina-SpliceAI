#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=drop0.2
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI/slurm/dropout0.2.out

#cd /home/scripts/

for i in {1..5}; 
    do python3 train_model.py $i retina dropout train --dropoutrate 0.2 > ../output_train/SpliceAI_dropout0.2_retina${i}.txt;
done
