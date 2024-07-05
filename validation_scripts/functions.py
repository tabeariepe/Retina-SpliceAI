import matplotlib.pyplot as plt
import re
from pkg_resources import resource_filename
from keras.models import load_model
import logging
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import tensorflow_addons as tfa


def plot_accuracy(file, model_num):
    with open(file) as f:
        text = f.read()

    # Define a regular expression pattern to match the training set metrics
    pattern = re.compile(r'set metrics:(.*?)Learning rate:', re.DOTALL)

    # Find all matches in the text
    matches = re.findall(pattern, text)

    acceptor_val = []
    donor_val = []
    acceptor_train = []
    donor_train = []

    # Extract and print the training set metrics
    for match in matches:
        all =  match.split(':')
        acceptor_val.append(all[3].strip())
        donor_val.append(all[14].strip())
        acceptor_train.append(all[26].strip())
        donor_train.append(all[37].strip())

    acceptor_val = [float(entry) for entry in acceptor_val]
    donor_val = [float(entry) for entry in donor_val]
    acceptor_train = [float(entry) for entry in acceptor_train]
    donor_train = [float(entry) for entry in donor_train]

    x = [i for i in range(1,11)]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(x, acceptor_val, label = 'acceptor_val', color = '#9FCAE6', marker='o')
    plt.plot(x, donor_val, label = 'donor_val', color = '#2E5B88', marker='o')
    plt.plot(x, acceptor_train, label = 'acceptor_train ', color = '#A6C48A', marker='o')
    plt.plot(x, donor_train, label = 'donor_train', color = '#678D58',  marker='o')
    plt.legend(loc = 'lower right') 
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training accuracy Model ' + str(model_num))
    plt.show()

    print(acceptor_val)

def plot_accuracy_multiple(file, model_num, color, ax=None):
    with open(file) as f:
        text = f.read()

    # Define a regular expression pattern to match the training set metrics
    pattern = re.compile(r'set metrics:(.*?)Learning rate:', re.DOTALL)

    # Find all matches in the text
    matches = re.findall(pattern, text)

    acceptor_val = []
    donor_val = []
    acceptor_train = []
    donor_train = []

    # Extract and print the training set metrics
    for match in matches:
        all =  match.split(':')
        acceptor_val.append(float(all[3].strip()))
        donor_val.append(float(all[14].strip()))
        acceptor_train.append(float(all[26].strip()))
        donor_train.append(float(all[37].strip()))

    x = [i for i in range(1, 11)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, acceptor_val, label='acceptor_val ' + str(model_num), color=color, marker='o')
    ax.plot(x, donor_val, label='donor_val ' + str(model_num), color=color, marker='o', linestyle='dashed')
    ax.plot(x, acceptor_train, label='acceptor_train ' + str(model_num), color=color, marker='o', linestyle='dotted')
    ax.plot(x, donor_train, label='donor_train ' + str(model_num), color=color, marker='o', linestyle='dashdot')

    return ax


class Annotator:

    def __init__(self, ref_fasta, annotations, model):

        # Decide which genome is used
        if annotations == 'grch37':
            annotations = resource_filename(__name__, '../annotations/grch37.txt')
        elif annotations == 'grch38':
            annotations = resource_filename(__name__, '../annotations/grch38.txt')
        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy()+1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])+1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        paths = ('../models/{}_{}.h5'.format(model, x) for x in range(1, 6))
        self.models = [load_model(resource_filename(__name__, x), custom_objects={'categorical_crossentropy_2d': categorical_crossentropy_2d}) for x in paths]


    def get_name_and_strand(self, chrom, pos):

        chrom = normalise_chrom(chrom, list(self.chroms)[0])
        idxs = np.intersect1d(np.nonzero(self.chroms == chrom)[0],
                              np.intersect1d(np.nonzero(self.tx_starts <= pos)[0],
                              np.nonzero(pos <= self.tx_ends)[0]))

        if len(idxs) >= 1:
            return self.genes[idxs], self.strands[idxs], idxs
        else:
            return [], [], []

    def get_pos_data(self, idx, pos):

        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        dist_exon_bdry = min(np.union1d(self.exon_starts[idx], self.exon_ends[idx])-pos, key=abs)
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)

        return dist_ann


def one_hot_encode(seq):

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]

def normalise_chrom(source, target):

    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source.strip('chr')
    elif not has_prefix(source) and has_prefix(target):
        return 'chr'+source

    return source


def get_score_position(chrom, pos, ann, dist_var):

    cov = 2*dist_var+1
    wid = 10000+cov
    delta_scores = []

    (genes, strands, idxs) = ann.get_name_and_strand(chrom, pos)
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

    import os
    # Limit the number of CPU threads globally for Keras
    os.environ["OMP_NUM_THREADS"] = "20"
        
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


def categorical_crossentropy_2d(y_true, y_pred):
    # Standard categorical cross entropy for sequence outputs

    return - kb.mean(y_true[:, :, 0]*kb.log(y_pred[:, :, 0]+1e-10)
                   + y_true[:, :, 1]*kb.log(y_pred[:, :, 1]+1e-10)
                   + y_true[:, :, 2]*kb.log(y_pred[:, :, 2]+1e-10))

def get_delta_scores(record, ann, dist_var, mask):

    cov = 2*dist_var+1
    wid = 10000+cov
    delta_scores = []

    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return delta_scores

    (genes, strands, idxs) = ann.get_name_and_strand(record.chrom, record.pos)
    if len(idxs) == 0:
        return delta_scores

    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][record.pos-wid//2-1:record.pos+wid//2].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return delta_scores

    if seq[wid//2:wid//2+len(record.ref)].upper() != record.ref:
        logging.warning('Skipping record (ref issue): {}'.format(record))
        return delta_scores

    if len(seq) != wid:
        logging.warning('Skipping record (near chromosome end): {}'.format(record))
        return delta_scores

    if len(record.ref) > 2*dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return delta_scores

    for j in range(len(record.alts)):
        for i in range(len(idxs)):

            if '.' in record.alts[j] or '-' in record.alts[j] or '*' in record.alts[j]:
                continue

            if '<' in record.alts[j] or '>' in record.alts[j]:
                continue

            if len(record.ref) > 1 and len(record.alts[j]) > 1:
                delta_scores.append("{}|{}|.|.|.|.|.|.|.|.".format(record.alts[j], genes[i]))
                continue

            dist_ann = ann.get_pos_data(idxs[i], record.pos)
            pad_size = [max(wid//2+dist_ann[0], 0), max(wid//2-dist_ann[1], 0)]
            ref_len = len(record.ref)
            alt_len = len(record.alts[j])
            del_len = max(ref_len-alt_len, 0)

            x_ref = 'N'*pad_size[0]+seq[pad_size[0]:wid-pad_size[1]]+'N'*pad_size[1]
            x_alt = x_ref[:wid//2]+str(record.alts[j])+x_ref[wid//2+ref_len:]

            x_ref = one_hot_encode(x_ref)[None, :]
            x_alt = one_hot_encode(x_alt)[None, :]

            if strands[i] == '-':
                x_ref = x_ref[:, ::-1, ::-1]
                x_alt = x_alt[:, ::-1, ::-1]

            y_ref = np.mean([ann.models[m].predict(x_ref) for m in range(5)], axis=0)
            y_alt = np.mean([ann.models[m].predict(x_alt) for m in range(5)], axis=0)

            if strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            if ref_len > 1 and alt_len == 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2+alt_len],
                    np.zeros((1, del_len, 3)),
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)
            elif ref_len == 1 and alt_len > 1:
                y_alt = np.concatenate([
                    y_alt[:, :cov//2],
                    np.max(y_alt[:, cov//2:cov//2+alt_len], axis=1)[:, None, :],
                    y_alt[:, cov//2+alt_len:]],
                    axis=1)            

            y = np.concatenate([y_ref, y_alt])

            idx_pa = (y[1, :, 1]-y[0, :, 1]).argmax()
            idx_na = (y[0, :, 1]-y[1, :, 1]).argmax()
            idx_pd = (y[1, :, 2]-y[0, :, 2]).argmax()
            idx_nd = (y[0, :, 2]-y[1, :, 2]).argmax()

            mask_pa = np.logical_and((idx_pa-cov//2 == dist_ann[2]), mask)
            mask_na = np.logical_and((idx_na-cov//2 != dist_ann[2]), mask)
            mask_pd = np.logical_and((idx_pd-cov//2 == dist_ann[2]), mask)
            mask_nd = np.logical_and((idx_nd-cov//2 != dist_ann[2]), mask)

            delta_scores.append("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                                record.alts[j],
                                genes[i],
                                (y[1, idx_pa, 1]-y[0, idx_pa, 1])*(1-mask_pa),
                                (y[0, idx_na, 1]-y[1, idx_na, 1])*(1-mask_na),
                                (y[1, idx_pd, 2]-y[0, idx_pd, 2])*(1-mask_pd),
                                (y[0, idx_nd, 2]-y[1, idx_nd, 2])*(1-mask_nd),
                                idx_pa-cov//2,
                                idx_na-cov//2,
                                idx_pd-cov//2,
                                idx_nd-cov//2))

    return delta_scores