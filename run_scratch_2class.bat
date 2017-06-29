set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\squad\entail_train_same_0_f1_2class.tsv --dev_path D:\users\t-yicxu\data\squad\entail_dev_same_0_f1_2class.tsv --test_path D:\users\t-yicxu\data\squad\entail_dev_same_0_f1_2class.tsv --word_vec_path D:\users\t-yicxu\data\squad\word2vec_nounk.txt --suffix squad_scratch_2class_nounk_f1 --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\squad --batch_size 60 --MP_dim 20 --with_highway --with_match_highway --with_aggregation_highway --aggregation_lstm_dim 300 