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
def get_predictions(exon_list, models, ss):

    reference = '../annotations/hg38.fa'
    annotation = '../annotations/combined.txt'

    annotators = []

    for model in models:
        annotators.append(Annotator(reference,annotation, model))

    result = []

    for i in exon_list:
        chr, strand, position = i
        scores = []
        for annotator in annotators:
            scores.append(get_score_position(chr, position, annotator, 10000))

        print(scores)

        if ss == 'acceptor':
            result.append([chr, position, strand, np.around(scores[0][1], decimals=2), np.around(scores[1][1], decimals=2), np.around(scores[2][1], decimals=2)])
        elif ss == 'donor':
            result.append([chr, position, strand, np.around(scores[0][2], decimals=2), np.around(scores[1][2], decimals=2), np.around(scores[2][2], decimals=2)])
        else:
            print('Splice site must be donor or acceptor')

    result_df = pd.DataFrame(result, columns=['chr', 'position', 'strand', 'retina','gtex', 'gtex2'])
    return result_df
# %%
# load the exons
sas_retina = pd.read_csv('../ref_data/sas_not_in_train.tsv', sep = '\t')
sas_retina = sas_retina.values.tolist()

sds_retina = pd.read_csv('../ref_data/sds_not_in_train.tsv', sep = '\t')
sds_retina = sds_retina.values.tolist()

# %%
models = ['SpliceAI_dropout_freeze_retina_all','SpliceAI_dropout0.3_gtex_all', 'SpliceAI_standard_gtex']

sas_prediction = get_predictions(sas_retina, models, 'acceptor')
sas_prediction.to_csv('../predictions/retina_sas_predictions.tsv', sep = '\t', index = False)
# %%
sds_prediction = get_predictions(sds_retina, models, 'donor')
sds_prediction.to_csv('../predictions/retina_sds_predictions.tsv', sep = '\t', index = False)

# %%
