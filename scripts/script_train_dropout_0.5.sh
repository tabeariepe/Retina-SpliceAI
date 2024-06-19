#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=drop0.5
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI/slurm/dropout0.5.out

#cd /home/scripts/

for i in {1..5}; 
    do python3 train_model.py $i retina dropout train --dropoutrate 0.5 > ../output_train_new/SpliceAI_dropout0.5_retina${i}.txt;
done
