set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\squad\entail_train_0_1_f1_2class_sentOnly.tsv --dev_path D:\users\t-yicxu\data\squad\entail_dev_0_1_f1_2class_sentOnly.tsv --test_path D:\users\t-yicxu\data\squad\entail_dev_0_1_f1_2class_sentOnly.tsv --word_vec_path D:\users\t-yicxu\data\squad\word2vec_nounk.txt --suffix squad_scratch_2class_nounk_sentOnly_new --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\squad --batch_size 60 --MP_dim 20 --with_highway --with_match_highway --with_aggregation_highway --aggregation_lstm_dim 100 --context_layer_num 2 --aggregation_layer_num 2 


