set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=1
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_replace_temp.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_replace_temp.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_replace_temp.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_temp.txt --suffix race_replace_test --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 2 --aggregation_lstm_dim 20 --max_sent_length 100 --use_options --wo_sort_instance_based_on_length --learning_rate 0.01 --max_epochs 30


