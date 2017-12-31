#python lemmatize_snts.py --data_dir ./full_data --lemma_dir ./lemmas --num_files 8
#python lemmatize_snts.py --data_dir ./eval_data --lemma_dir ./lemmas --num_files 1
python lemmatize_snts.py --data_dir $1 --lemma_dir ./lemmas --num_files 1
