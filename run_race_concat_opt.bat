set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_concat_options_all.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_concat_options_all.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_concat_options_all.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_all.txt --suffix race_concat_opt_all --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 10 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 500 --use_options --wo_sort_instance_based_on_length --learning_rate 0.001


