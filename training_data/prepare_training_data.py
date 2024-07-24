# Import
import pandas as pd
import pysam
import numpy as np
import utils
import itertools
import matplotlib.pyplot as plt
import sys

# Parameters
star_tab_file = sys.argv[1]
min_samples = int(sys.argv[2])
outfile = sys.argv[3]
reference_genes = '../ref_data/Homo_sapiens.GRCh38.110.all_known_genes.sorted.formatted.bed.gz'
biomart_paralogs = "../ref_data/mart_export_paralogs.txt"

# Load TSS and TES for prinicipal transcripts
ref_tss_tes = pd.read_csv('../ref_data/tss_tes.txt', sep = '\t')

# filter for the first txstart and last txend of each gene 
tss_tes_dict = {}
strand_dict = {}
for index,row in ref_tss_tes.iterrows():
    gene = row['Gene name']
    start = row['Gene start (bp)']
    end = row['Gene end (bp)']
    if gene in tss_tes_dict.keys():
        if start < tss_tes_dict[gene][0]:
            tss_tes_dict[gene][0] = start
        if end > tss_tes_dict[gene][1]:
            tss_tes_dict[gene][1] = end
    else:
        tss_tes_dict[gene] = [start, end]
        strand_dict[gene] = row['Strand']

# Load the STAR splice junctions
star = pd.read_csv(star_tab_file, sep = '\t', names = ['chr','start', 'end', 'strand','intron','annotated','number_of_reads', 'multimapping', 'overhang'], low_memory=False)
# remove junctions on mitochondrial chromosomes 
star = star[star['chr'] != 'MT']

# Group the juncctions, add a number of samples column, and combine the number of reads of the different samples
result_df = star.groupby(['chr', 'start', 'end', 'strand', 'intron', 'annotated']).agg(samples=('number_of_reads', 'size'), reads=('number_of_reads', 'sum')).reset_index()
result_df = result_df[result_df['samples'] >= min_samples]

# Add a gene to each splice junction
genes = pysam.TabixFile(reference_genes)
result_df['genes'] = result_df.apply(lambda x: utils.gencode_all_known_genes(x[['chr', 'start', 'end']], genes), axis=1)

# Drop splice junctions that are not mapped to any known gene 
a = len(result_df.index)
df = result_df[result_df['genes'] != 'NA(0)']

# get genes in right format
for index,row in df.iterrows():

    genes = row['genes'].split(';')
    
    if len(genes) == 1:
        df.at[index,'genes'] = [genes[0].split('(')[0]]
    else:
        
        genenames = [i.split('(')[0] for i in genes]
        percentages = [float(i.split('(')[1][:-1]) for i in genes]
        
        max_value = max(percentages)

        positions = [i for i, j in enumerate(percentages) if j == max_value]
        if len(positions) == 1:
            df.at[index,'genes'] = [genenames[positions[0]]]
        else:
            df.at[index,'genes'] = [genenames[i] for i in positions]

# Only keep junctions for wich the junction and gene are on the same strand
# Replace -1 with 2
for key, value in strand_dict.items():
    if value == -1:
        strand_dict[key] = 2

df['match'] = pd.Series(dtype=object)

for index, row in df.iterrows():
    match = []
    for gene in row['genes']:
        if gene in strand_dict:
            match.append(row['strand'] == strand_dict[gene])
        else:
            match.append(False)
    df.at[index, 'match'] = match


d = dict()

for index,row in df.iterrows():
     
    gene = row['genes']
    match = row['match']
        
    for g,m in zip(gene,match):
        # only keep exons that are ont he same strand as the gene that they are assigned to 
        if m == True:
            if g in d.keys():
                d[g][2].append(row['start'])
                d[g][3].append(row['end'])
            else:
                d[g] = [row['chr'], row['strand'], [row['start']], [row['end']]]


#look up gene paralogs
paralogs_df = pd.read_csv(biomart_paralogs)

no_paralog = paralogs_df[paralogs_df['Human paralogue associated gene name'].isna()]
paralogs_df = paralogs_df.dropna()
biomart_paralogs = paralogs_df['Gene name'].tolist()
biomart_noparalog = no_paralog['Gene name'].tolist()
biomart_paralogs = list(set(biomart_paralogs))

# Convert the splice junctions to a df 
spliceaidf = pd.DataFrame.from_dict(d, orient = 'index', columns=['chr', 'strand', 'startsites', 'endsites'])
spliceaidf.reset_index(level=0, inplace=True)

# Add TSS and TES
# Remove genes that are not in the canonical dataset
spliceaidf = spliceaidf[spliceaidf['index'].isin(tss_tes_dict.keys())]

spliceaidf['TES'] = ''
spliceaidf['TSS'] = ''
spliceaidf['paralog'] = ''

for index, row in spliceaidf.iterrows():
    tss, tes = tss_tes_dict[row['index']]
    spliceaidf.at[index, 'TSS'] = int(tss)
    spliceaidf.at[index, 'TES'] = int(tes)

    # Reformat intron junctions to exon junctions and keep unique junctions
    spliceaidf.at[index, 'startsites'] = list({int(i) - 1 for i in row['startsites'] if i > tss and i < tes})
    spliceaidf.at[index, 'endsites'] = list({int(i) + 1 for i in row['endsites'] if i > tss and i < tes})

    # Sort the exon start and end sites
    spliceaidf.at[index, 'startsites'] = sorted(spliceaidf.at[index, 'startsites'])
    spliceaidf.at[index, 'endsites'] = sorted(spliceaidf.at[index, 'endsites'])

    # add paralog information
    if row['index'] in biomart_paralogs:
        spliceaidf.at[index, 'paralog'] = 1
    else: 
        spliceaidf.at[index, 'paralog'] = 0

# Drop genes without start or endsites
spliceaidf = spliceaidf[(spliceaidf['startsites'].apply(lambda x: len(x) != 0)) & (spliceaidf['endsites'].apply(lambda x: len(x) != 0))]

# Create the output file
with open (outfile, 'w') as f:

    for index,row in spliceaidf.iterrows():
        gene = row['index']
        paralog = int(row['paralog'])
        chromosome = row['chr']
        tss = row['TSS']
        tes = row['TES']
        exon_end = str(row['endsites'])[1:-1].replace(' ', '')
        exon_start = str(row['startsites'])[1:-1].replace(' ', '')

        if row['strand'] == 1:
            strand = '+'
        else:
            strand = '-' 

        f.write(gene + '\t' + str(paralog) + '\t' + str(chromosome) + '\t' + strand + '\t' + str(tss) + '\t' + str(tes) + '\t' + exon_start + ',\t' + exon_end + ',\n')