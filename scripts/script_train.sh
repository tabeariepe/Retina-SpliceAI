
for i in {1..5};
    do python3 train_model.py $i retina standard train > ../output_train_new/SpliceAI_standard_retina${i}.txt;
done
