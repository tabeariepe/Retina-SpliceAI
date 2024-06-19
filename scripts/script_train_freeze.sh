#!/bin/bash

for option in A B C D E F; do
    for i in {1..5}; do
        python3 train_model.py $i retina freeze train --freezeoption $option > ../output_train/SpliceAI_freeze${option}_retina${i}.txt
    done
done
