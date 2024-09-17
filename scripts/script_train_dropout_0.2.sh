#!/bin/bash

for i in {1..5};
  do python3 train_model.py $i retina dropout train --dropoutrate 0.2 > ../output_train_new/SpliceAI_dropout0.2_new_retina${i}.txt;
done
