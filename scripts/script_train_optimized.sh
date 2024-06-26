#!/bin/bash
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=5-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=optimized
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI/slurm/optimized.out

#cd /home/scripts/
 
python3 train_model.py 4 retina optimized train > ../output_train_new/SpliceAI_optimized_retina4.txt;