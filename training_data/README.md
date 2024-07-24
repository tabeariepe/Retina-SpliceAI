## RNA-sequencing data

### Download FASTQ files

Accession Pinelli et al.: PRJEB42859
Accesion Ratnapriya et al: PRJNA476171

```
java -jar ena-file-downloader.jar \
--accessions=number --format=READS_FASTQ \
--location=./ --protocol=FTP asperaLocation=null \
--email=name@mail.com
```

### Analysis

**Reference genome**
ENSEMBL release 110 (downloaded on 08/01/2024)

**TrimGalore (v.0.6.10)**
Trimming failed for two files (SRR7460904 and SRR7461157), so those are excluded

```
../tools/TrimGalore-0.6.10/trim_galore --fastqc --paired --illumina --cores 20 \
-o training_data/pinelli/trimmed \
--path_to_cutadapt ../tools/cutadapt \
$file1 $file2 \
```

**STAR (v.2.7.11a)**

Create a STAR index
```
../tools/STAR-2.7.11a/bin/Linux_x86_64/STAR \
 --runThreadN 20 --runMode genomeGenerate \
--genomeDir star_index \
--genomeFastaFiles "../ref_data/Homo_sapiens.GRCh38.dna.primary_assembly.fa" \
--sjdbGTFfile "../ref_data/Homo_sapiens.GRCh38.110.gtf" \
--sjdbOverhang 125
```

Run STAR on all samples together
```
 $STAR_PATH \
  --genomeDir $GENOME_DIR \
  --runThreadN 12 \
  --readFilesCommand zcat \
  --readFilesIn $file1 $file2 \
  --outFileNamePrefix $OUTPUT_DIR$unique_identifier. \
  --outSAMtype BAM SortedByCoordinate \
  --twopassMode Basic \
```

### Prepare the SpliceAI training data

Combine the STAR splice junctinos from all samples in one file:
-  Filters for column 5 > 0, which means that we only consider canonical splice junctions
- Filters for column 7 > 2, which means that each junction has to have at least 3 uniquly mapped reads 
- Filter for column 9 > 4, which means the maximum spliced alignment overhang is at least 5 nucleotides

```
cat *.SJ.out.tab | awk '($5 > 0 && $7 > 2 && $9 > 4 )' | sort > SJ.filtered2.tab
```

Run `prepare_training_data.py` to convert the STAR splice junctions into SpliceAI training data format. 

### Create the SpliceAI training data
```
python create_datafile.py train all
python create_datafile.py test 0
# This caused an error for gene U3 so I excluded it from the analysis

python create_dataset.py train all
python create_dataset.py test 0
```

Additionally, we created a train dataset with all chromosomes for both the retina and GTEx dataset:

```
python create_datafile.py all all
python create_dataset.py all all
```