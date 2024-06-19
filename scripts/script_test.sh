#!/bin/bash

#GTEx
python3 test_model.py SpliceAI_standard_gtex retina > ../output_test/SpliceAI_standard_gtex.txt
python3 test_model.py SpliceAI_standard_gtex gtex >> ../output_test/SpliceAI_standard_gtex.txt
python3 test_model.py SpliceAI_standard_gtex canonical >> ../output_test/SpliceAI_standard_gtex.txt