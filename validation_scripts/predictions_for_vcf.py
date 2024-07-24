import pysam
from functions import *
import argparse 

print('Import succesful')

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('input', type=str, help='Input file')
parser.add_argument('output', type=str, help='Output file')

# Parse the arguments
args = parser.parse_args()

vcf_file = args.input
output_file = args.output

reference = '../annotations/hg38.fa'
annotation = '../annotations/combined.txt'

vcf = pysam.VariantFile(vcf_file)

retina = Annotator(reference, annotation, 'SpliceAI_dropout_freeze_retina_all')
gtex = Annotator(reference, annotation, 'SpliceAI_dropout0.3_gtex_all')
result = []
for record in vcf:
    retina_score = get_delta_scores(record, retina, 10000, 0)
    gtex_score = get_delta_scores(record, gtex, 10000, 0)
    result.append([record.pos, retina_score, gtex_score])

with open (output_file, 'w') as file:
    for i in result:
        file.write(str(i) + '\n')