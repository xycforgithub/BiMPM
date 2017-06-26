set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2,3
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\snli_1.0\train_2class.tsv --dev_path D:\users\t-yicxu\data\snli_1.0\dev_2class.tsv --test_path D:\users\t-yicxu\data\snli_1.0\test_2class.tsv --word_vec_path D:\users\t-yicxu\data\snli_1.0\word2vec.txt --suffix snli_new --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\ --batch_size 60 --MP_dim 20 --with_highway --with_match_highway --with_aggregation_highway --aggregation_lstm_dim 300 