#!/bin/bash
#$ -l h_rt=168:00:00 -l rmem=8G -m bea -M ziqi.zhang@sheffield.ac.uk

module load apps/python/anaconda3-4.2.0
source activate msm4phi
export PYTHONPATH=/home/li1zz/msm4phi/code/python/src

oauth=
keywords=
solr=

python3 -m data.twitter_collector ${oauth} ${keywords} ${solr}

