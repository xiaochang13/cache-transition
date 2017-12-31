#!/bin/bash
#python -u ./oracle.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel --conll_file train.conll
#python -u ./oracle.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel --conll_file simple.conll 
#python -u ./oracle.py --data_dir ../../dev --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel --conll_file dev.conll --dep_file ./dev/dep.token
python -u ./oracle.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel --conll_file train.conll --dep_file ./train/dep.token
#python -u ./oracle.py --data_dir ../../dev --use_lemma --run_dir dev_dir --stats_dir stats --use_stats --parallel --conll_file dev.conll --dep_file ./dev_parallel/dep.token
