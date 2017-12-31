#!/bin/bash
#python -u ./oracle.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel --conll_file train.conll
python -u ./treewidth.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --parallel
#python -u ./treewidth.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --high_width_file reversed_highwidth.txt --parallel --reversed > reversed_width.txt
#python -u ./treewidth.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --high_width_file random_highwidth.txt --parallel --random > random_width.txt
#python ./categorize_amr.py --data_dir ../../dev --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
#python ./categorize_amr.py --data_dir ../../eval --use_lemma --run_dir run_dir --stats_dir stats --use_stats --map_file ./run_dir/train_map
#python -u ./treewidth.py --data_dir ../../train --use_lemma --run_dir run_dir --stats_dir stats --use_stats --high_width_file depth_highwidth.txt --parallel --depth > depth_width.txt
#python -u ./treewidth.py --data_dir ./AMR-aligned/dev --run_dir run_dir --stats_dir stats --use_stats --high_width_file dev_highwidth.txt --parallel --depth > dev_width.txt
#python -u ./treewidth.py --data_dir ./AMR-aligned/test --run_dir run_dir --stats_dir stats --use_stats --high_width_file test_highwidth.txt --parallel --depth > test_width.txt
