#!/bin/bash

pip3 install pyfaidx

cd /home/validation_scripts/
python3 predictions_retina_and_control_exons.py > exons.txt
