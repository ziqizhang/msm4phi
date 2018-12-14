#!/bin/bash
#$ -l h_rt=168:00:00 -l rmem=8G -m bea -M ziqi.zhang@sheffield.ac.uk

#module load apps/python/anaconda3-4.2.0
#source activate msm4phi
export PYTHONPATH=/home/zz/Work/msm4phi/code/python/src

oauth=/home/zz/Work/msm4phi/resources/config/twitter_oauth_zqzuk.txt
keywords=/home/zz/Work/msm4phi/resources/config/tag_list_mu
solr=http://localhost:8983/solr

python3 -m data.twitter_collector ${oauth} ${keywords} ${solr}

