#!/bin/bash

for i in {1..5}; do
    python3 train_model.py $i retina freeze train --freezeoption none > ../output_train_new/SpliceAI_freeze_none_retina${i}.txt
done
