#!/bin/bash
#python ./categorize_amr.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel
#python ./categorize_amr.py --data_dir ../../dev --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
#python ./categorize_amr.py --data_dir ../../eval --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
#python ./prepareTokens.py --data_dir ../../dev --use_lemma --run_dir new_dev_run --stats_dir stats --use_stats --parallel --conll_file dev.conll
#python ./prepareTokens.py --data_dir ../../train --use_lemma --run_dir train --stats_dir stats --use_stats --conll_file train.conll --dev_dir ../../dev --dev_output dev --test_dir ../../test --test_output test
#mkdir -p dev_categorized
mkdir -p dev_temp
#python ./prepareTokens.py --data_dir ../../train --use_lemma --run_dir train --stats_dir stats --use_stats --conll_file temp.conll --dev_dir previous/dev_tokenized --dev_output dev_temp --test_dir ../../test --test_output test
python ./prepareTokens.py --data_dir ../../train --use_lemma --run_dir train --stats_dir stats --use_stats --conll_file temp.conll --dev_dir previous/dev_tokenized --dev_output dev_output --test_dir ./test_tokenized --test_output test_output
#python ./prepareTokens.py --realign --data_dir ../../train --use_lemma --run_dir train --stats_dir stats --use_stats --conll_file train.conll --dev_dir dev_tokenized --dev_output dev --test_dir ../../test --test_output test --dep_file ./train_categorized/tok_0
