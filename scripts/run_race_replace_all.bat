set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_replace_1_all_shuffled.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_replace_1_all_shuffled.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_replace_1_middle_shuffled.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_all.txt --suffix race_replace_all --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 10 --max_sent_length 500 --learning_rate 0.001 --display_every 1 --with_highway --with_match_highway --with_aggregation_highway


