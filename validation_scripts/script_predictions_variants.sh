#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --container-mounts=/home/tabeariepe/Retina-SpliceAI-2/:/home
#SBATCH --container-image="doduo1.umcn.nl#tabea/spliceai:2.3"
#SBATCH --job-name=variants2
#SBATCH --exclude=dlc-electabuzz
#SBATCH --output=/home/tabeariepe/Retina-SpliceAI-2/slurm/variants2.out

pip3 install pysam
pip3 install pyfaidx

cd /home/validation_scripts/

#python3 predictions_for_vcf.py ../variants/retina_specific_hg38.vcf ../predictions/retina_specific_variants.tsv > retina_specific.txt
#python3 predictions_for_vcf.py ../variants/unsolved_retnet.vcf ../predictions/unsolved_retnet.tsv >> retina_specific.txt
#python3 predictions_for_vcf.py ../variants/merged_filtered_5.vcf.gz ../predictions/unsolved_all_5.tsv > all_variants.txt

#python3 predictions_for_vcf.py ../variants/split.chr1.vcf.gz ../predictions/variants_chr1.tsv > chr1.txt
python3 predictions_for_variants.py ../variants/split.chr21.vcf.gz ../predictions/variants_chr21.tsv > chr21.txt