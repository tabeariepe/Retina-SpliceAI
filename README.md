# Retina-SpliceAI
This repository contains the scripts used for the paper '_Deep learning-based prediction of tissue-specific splice sites in the human neural retina' by Riepe et al. (2024).

## Data
In [Training data](training_data) we describe how we obtained the retina dataset included in [datasets](datasets). 

## Code
- [Figures and tables](figures_and_tables) contains all scripts used to create the figures and tables for the manuscript
- [Models](models) contains all models that we trained. For each hyperparameter setting, five separate models were trained.
- [Output test](output_test) contains the test output for each model. Each model was evaluated using the retina, GTEx, and GENCODE (canonical) dataset. 
- [Output_train](output_train) contains the training output for each model.
- [Predictions](predictions) contains the predictions for the retina-enriched exons, control exons, and variants with a retina-specific splicing defect.
- [Reference data](ref_data) contains all reference files used for validation of the models. This includes the retina-enriched exons and control exons.
- [Scripts](scripts) contains all scripts used for training and testing the model.
- [Validation scripts](validation_scripts) contains the scripts used for validating the models on the exons and variants.
- [Variants](variants) contains the vcf file of the variants with a retina-specific splicing defect.
