#!/bin/bash
DATA_DIR=../../AMRParser/preprocessing/seq2graph/amr2seq/data_prep/train_categorized
OUTPUT_DIR=oracle_output
python ./oracle.py --data_dir $DATA_DIR --output_dir $OUTPUT_DIR
