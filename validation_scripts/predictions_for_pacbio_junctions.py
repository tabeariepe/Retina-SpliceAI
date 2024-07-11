from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from functions import *

def get_score_position(chrom, pos, ann, dist_var):

    cov = 2*dist_var+1
    wid = 10000+cov
    delta_scores = []

    (_, strands, idxs) = ann.get_name_and_strand(chrom, pos)
    if len(idxs) == 0:
        return delta_scores

    chrom = normalise_chrom(chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][pos-wid//2-1:pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(pos))
        return delta_scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(pos))
        return delta_scores
        
    for i in range(len(idxs)):

        dist_ann = ann.get_pos_data(idxs[i], pos)
        pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]

        x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]

        x_ref = one_hot_encode(x_ref)[None, :]

        if strands[i] == '-':
            x_ref = x_ref[:, ::-1, ::-1]

        y_ref = np.mean([ann.models[m].predict(x_ref, verbose=0) for m in range(len(ann.models))], axis=0)

        if strands[i] == '-':
            y_ref = y_ref[:, ::-1]

    return y_ref[0,y_ref.shape[1]//2]

# Define the models that the script should run on
models = ['SpliceAI_dropout_freeze_retina_all', 'SpliceAI_dropout0.3_gtex_all']

# Load the novel acceptor and donor sites
acceptors = pd.read_csv('../ref_data/pacbio_novel_acceptors.tsv', sep = '\t', header = None)
donors = pd.read_csv('../ref_data/pacbio_novel_donors.tsv', sep = '\t', header = None)
acceptors = acceptors.values.tolist()
donors = donors.values.tolist()
print('Number of novel acceptors: ', len(acceptors))
print('Number of novel donors: ', len(donors))

# Define the reference genome and retina (ENSEMBL + PacBio) combined annotation
reference = '../annotations/hg38.fa'
annotation = '../annotations/combined.txt'

annotators = []
for m in models:
    annotators.append(Annotator(reference,annotation, m))

columns = ['chr', 'start'] + [model for model in models]
print(columns)

# Get the scores for the splice acceptor sites
result_acceptors = []

for i in acceptors:
    chr,position = i

    scores = []

    for a in annotators:
        
        scores.append(get_score_position(chr, position, a, 10000))

    result_acceptors.append([chr, position] + [np.around(scores[i], decimals=2) for i in range(len(scores))])

columns = ['chr', 'start'] + [model for model in models]
print(columns)
result_acceptors_df = pd.DataFrame(result_acceptors, columns=columns)
result_acceptors_df.to_csv('../predictions/pacbio_acceptors_predictions.tsv', sep = '\t', index = False)

result_donors = []

for i in donors:
    chr,position = i

    scores = []

    for a in annotators:
        scores.append(get_score_position(chr, position, a, 10000))

    result_donors.append([chr, position] + [np.around(scores[i], decimals=2) for i in range(len(scores))])

result_donors_df = pd.DataFrame(result_donors, columns=columns)
result_donors_df.to_csv('../predictions/pacbio_donors_predictions.tsv', sep = '\t', index = False)