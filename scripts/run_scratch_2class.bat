set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\squad\entail_train_4_0_f1_2class_re3.tsv --dev_path D:\users\t-yicxu\data\squad\entail_dev_1_0_f1_2class_re1.tsv --test_path D:\users\t-yicxu\data\squad\entail_dev_1_0_f1_2class_re1.tsv --word_vec_path D:\users\t-yicxu\data\squad\word2vec_entail_4_re3_devre1.txt --suffix squad_scratch_2class_nounk_4_re3_fixword --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\squad --batch_size 60 --MP_dim 10 --with_highway --with_match_highway --with_aggregation_highway --aggregation_lstm_dim 100 --context_layer_num 2 --aggregation_layer_num 2 


