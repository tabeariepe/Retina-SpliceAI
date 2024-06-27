#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --container-mounts=/data/temporary/tabea/Retina-SpliceAI-2-paper/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=test
#SBATCH --output=/data/temporary/tabea/Retina-SpliceAI-2-paper/slurm/test.out

cd /home/scripts/

#GTEx
# python3 test_model.py SpliceAI_standard_gtex retina > ../output_test/SpliceAI_standard_gtex.txt
# python3 test_model.py SpliceAI_standard_gtex gtex >> ../output_test/SpliceAI_standard_gtex.txt
# python3 test_model.py SpliceAI_standard_gtex canonical >> ../output_test/SpliceAI_standard_gtex.txt

# retina
# python3 test_model.py SpliceAI_standard_retina retina > ../output_test/SpliceAI_standard_retina.txt
# python3 test_model.py SpliceAI_standard_retina gtex >> ../output_test/SpliceAI_standard_retina.txt
# python3 test_model.py SpliceAI_standard_retina canonical >> ../output_test/SpliceAI_standard_retina.txt

#Freeze A
# python3 test_model.py SpliceAI_freezeA_retina retina > ../output_test/SpliceAI_freezeA_retina.txt
# python3 test_model.py SpliceAI_freezeA_retina gtex >> ../output_test/SpliceAI_freezeA_retina.txt
# python3 test_model.py SpliceAI_freezeA_retina canonical >> ../output_test/SpliceAI_freezeA_retina.txt

# #Freeze B
# python3 test_model.py SpliceAI_freezeB_retina retina > ../output_test/SpliceAI_freezeB_retina.txt
# python3 test_model.py SpliceAI_freezeB_retina gtex >> ../output_test/SpliceAI_freezeB_retina.txt
# python3 test_model.py SpliceAI_freezeB_retina canonical >> ../output_test/SpliceAI_freezeB_retina.txt

# #Freeze C
# python3 test_model.py SpliceAI_freezeC_retina retina > ../output_test/SpliceAI_freezeC_retina.txt
# python3 test_model.py SpliceAI_freezeC_retina gtex >> ../output_test/SpliceAI_freezeC_retina.txt
# python3 test_model.py SpliceAI_freezeC_retina canonical >> ../output_test/SpliceAI_freezeC_retina.txt

# #Freeze D
# python3 test_model.py SpliceAI_freezeD_retina retina > ../output_test/SpliceAI_freezeD_retina.txt
# python3 test_model.py SpliceAI_freezeD_retina gtex >> ../output_test/SpliceAI_freezeD_retina.txt
# python3 test_model.py SpliceAI_freezeD_retina canonical >> ../output_test/SpliceAI_freezeD_retina.txt

# #Freeze E
# python3 test_model.py SpliceAI_freezeE_retina retina > ../output_test/SpliceAI_freezeE_retina.txt
# python3 test_model.py SpliceAI_freezeE_retina gtex >> ../output_test/SpliceAI_freezeE_retina.txt
# python3 test_model.py SpliceAI_freezeE_retina canonical >> ../output_test/SpliceAI_freezeE_retina.txt

# #Freeze F
# python3 test_model.py SpliceAI_freezeF_retina retina > ../output_test/SpliceAI_freezeF_retina.txt
# python3 test_model.py SpliceAI_freezeF_retina gtex >> ../output_test/SpliceAI_freezeF_retina.txt
# python3 test_model.py SpliceAI_freezeF_retina canonical >> ../output_test/SpliceAI_freezeF_retina.txt

#Freeze none
python3 test_model.py SpliceAI_freezenone_retina retina > ../output_test/SpliceAI_freezenone_retina.txt
python3 test_model.py SpliceAI_freezenone_retina gtex >> ../output_test/SpliceAI_freezenone_retina.txt
python3 test_model.py SpliceAI_freezenone_retina canonical >> ../output_test/SpliceAI_freezenone_retina.txt