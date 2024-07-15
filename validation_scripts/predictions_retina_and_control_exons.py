# %%
# Imports 
from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from keras.models import load_model
from functions import *

print('Import succesful')

# Define a function to get the predictions
def get_predictions(exon_list, models):

    reference = '../annotations/hg38.fa'
    annotation = '../annotations/combined.txt'

    annotators = []

    for model in models:
        annotators.append(Annotator(reference,annotation, model))


    result = []

    for i in exon_list:
        chr,start,end, strand = i

        scores = []

        if strand == 1.0:
            for annotator in annotators:
                scores.append(get_score_position(chr, start, annotator, 10000))
                scores.append(get_score_position(chr, end, annotator, 10000))


        elif strand == -1.0:
            for annotator in annotators:
                scores.append(get_score_position(chr, end, annotator, 10000))
                scores.append(get_score_position(chr, start, annotator, 10000))

        else:
            print('Strand info is missing')

        result.append([chr, start, end, strand, np.around(scores[0], decimals=2), np.around(scores[2], decimals=2), np.around(scores[4], decimals=2),
                    np.around(scores[1], decimals=2), np.around(scores[3], decimals=2),  np.around(scores[5], decimals=2)])
        

    result_df = pd.DataFrame(result, columns=['chr', 'start', 'end', 'strand', 'retina_acceptor','gtex_acceptor', 'gtex2_acceptor', 'retina_donor', 'gtex_donor', 'gtex2_donor'])
    return result_df
# %%
# load the exons
short_exons = pd.read_csv('../ref_data/short_exons.bed', sep = '\t')
short_exons = short_exons.values.tolist()
long_exons = pd.read_csv('../ref_data/long_exons.bed', sep = '\t')
long_exons = long_exons.values.tolist()
musashi_exons = pd.read_csv('../ref_data/Musashi_hg38.bed', sep = '\t')
# %%
musash_exons = musashi_exons.values.tolist()
retina_exons = short_exons + long_exons + musash_exons
print('Number of retina-specific exons: ', len(retina_exons))
# %%
control_exons = pd.read_csv('../ref_data/matching_exons.bed', sep = '\t')
control_exons = control_exons.values.tolist()

models = ['SpliceAI_dropout_freeze_retina_all','SpliceAI_dropout0.3_gtex_all', 'SpliceAI_standard_gtex']

retina_prediction = get_predictions(retina_exons, models)
retina_prediction.to_csv('../predictions/retina_exons_predictions.tsv', sep = '\t', index = False)

control_prediction = get_predictions(control_exons, models)
control_prediction.to_csv('../predictions/control_exons_predictions.tsv', sep = '\t', index = False)