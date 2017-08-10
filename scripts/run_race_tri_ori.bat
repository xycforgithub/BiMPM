set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=3
set TF_CPP_MIN_LOG_LEVEL=1
python src\TriMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_concat_tri_middle_shuffled.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_concat_tri_middle.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_concat_tri_middle.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_tri_middle.txt --suffix race_tri_ori_middle --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 5 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 400 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 30 --max_hyp_length 80


