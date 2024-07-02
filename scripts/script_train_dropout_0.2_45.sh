#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI-2/:/home,/data/temporary/tabea/Retina-SpliceAI-2-paper/data/:/data
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=drop0.2_45
#SBATCH --exclude=dlc-electabuzz,dlc-groudon,dlc-lugia
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI-2/slurm/dropout0.2_45.out

cd /home/scripts/

for i in {4..5};
    do python3 train_model.py $i retina dropout train --dropoutrate 0.2 > ../output_train_new/SpliceAI_dropout0.2_new_retina${i}.txt;
done