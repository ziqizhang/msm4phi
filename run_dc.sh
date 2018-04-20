#!/bin/bash
#$ -l h_rt=168:00:00 -l rmem=12G -m bea -M ziqi.zhang@sheffield.ac.uk

module load apps/python/anaconda3-4.2.0
source activate msm4phi
export PYTHONPATH=/home/li1zz/msm4phi/code/python/src

input=/home/ac4zzh/chase/data/ml/ml/rm/labeled_data_all.csv
output=/home/ac4zzh/chase/output
emg_model=/home/ac4zzh/GoogleNews-vectors-negative300.bin.gz
emg_dim=300
emt_model=/home/ac4zzh/Set1_TweetDataWithoutSpam_Word.bin
emt_dim=300
data=rm
targets=2
word_norm=0
ug=0

