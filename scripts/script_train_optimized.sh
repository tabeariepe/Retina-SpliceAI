#!/bin/bash

for i in {1..5}; do 
  python3 train_model.py $i retina optimized train > ../output_train_new/SpliceAI_optimized_retina${i}.txt;
done
