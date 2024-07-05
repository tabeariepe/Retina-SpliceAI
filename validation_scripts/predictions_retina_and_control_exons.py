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

    reference = '/data/annotations/hg38.fa'
    annotation = '/data/annotations/combined.txt'

    annotators = []

    for model in models:
        annotators.append(Annotator(reference,annotation, model))


    result = []

    for i in exon_list:
        chr,start,end, strand = i

        scores = []

        if strand == 1.0:
            for model,annotator in zip(models,annotators):
                scores.append(get_score_position(chr, start, annotator, 10000, model))
                scores.append(get_score_position(chr, end, annotator, 10000, model))


        elif strand == -1.0:
            for model,annotator in zip(models,annotators):
                scores.append(get_score_position(chr, end, annotator, 10000, model))
                scores.append(get_score_position(chr, start, annotator, 10000, model))

        else:
            print('Strand info is missing')

        result.append([chr, start, end,  np.around(scores[0], decimals=2), np.around(scores[2], decimals=2), np.around(scores[4], decimals=2),
                       np.around(scores[6], decimals=2), np.around(scores[8], decimals=2),
                    np.around(scores[1], decimals=2), np.around(scores[3], decimals=2), np.around(scores[5], decimals=2),
                    np.around(scores[7], decimals=2), np.around(scores[9], decimals=2)])
        

    result_df = pd.DataFrame(result, columns=['chr', 'start', 'end', 'gtex_acceptor', 'pinelli_acceptor', 'optimized_acceptor', 'dropout0.2_acceptor', 'dropout0.4_acceptor',
                                              'gtex_donor', 'pinelli_donor', 'optimized_donor', 'dropout0.2_donor', 'dropout_0.4_donor'])
    return result_df

# load the exons

short_exons = pd.read_csv('../ref_data/short_exons.bed', sep = '\t')
short_exons = short_exons.values.tolist()
long_exons = pd.read_csv('../ref_data/long_exons.bed', sep = '\t')
long_exons = long_exons.values.tolist()
musashi_exons = pd.read_csv('../ref_data/Musashi_hg38.bed', sep = '\t')
musash_exons = musashi_exons.values.tolist()
retina_exons = short_exons + long_exons + musash_exons
control_exons = pd.read_csv('../ref_data/matching_exons.bed', sep = '\t')
control_exons = control_exons.values.tolist()

models = ['SpliceAI_dropout_freeze_retina_all','SpliceAI_standard_gtex']

retina_prediction = get_predictions(retina_exons, models)
retina_prediction.to_csv('../predictions/retina_exons_predictions.tsv', sep = '\t', index = False)

control_prediction = get_predictions(control_exons, models)
control_prediction.to_csv('../predictions/control_exons_predictions.tsv', sep = '\t', index = False)