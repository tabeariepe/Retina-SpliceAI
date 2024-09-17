#!/bin/bash
for i in {1..5};
  do python3 train_model.py $i gtex standard train > ../output_train_new/SpliceAI_standard_gtex${i}.txt;
done
