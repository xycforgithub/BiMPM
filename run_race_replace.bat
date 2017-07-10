set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=3
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_replace_3.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_replace.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_replace.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec.txt --suffix race_replace_3_fixword --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 10 --with_highway --with_match_highway --with_aggregation_highway --aggregation_lstm_dim 100 --max_sent_length 500 --learning_rate 0.01


