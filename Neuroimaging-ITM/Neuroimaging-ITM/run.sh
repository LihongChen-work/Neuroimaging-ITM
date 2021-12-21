#!/bin/bash

timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
output_dir='./logs/'
config_file='./configs/bio_config'
config_adv_file='./configs/bio_config_adv'

# unzip the embeddings file 解压缩嵌入文件
unzip data/CoNLL04/vecs.lc.over100freq.zip -d data/CoNLL04/

mkdir -p $output_dir


#在训练集上训练并在验证集上评估以获得早停周期
python -u train_es.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt
