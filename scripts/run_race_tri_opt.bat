set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2
set TF_CPP_MIN_LOG_LEVEL=1
rem python src\TriMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_concat_options_tri_all_sorted_part2.tsv --dev_path D:\users\t-yicxu\data\race\entail_train_concat_options_tri_all_sorted_part2.tsv --test_path D:\users\t-yicxu\data\race\entail_train_concat_options_tri_all_sorted_part2.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_tri_all.txt --suffix race_tri_opt_all_part2 --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 5 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 400 --use_options --wo_sort_instance_based_on_length --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 1 --max_hyp_length 80
python src\TriMatchTrainer.py --train_path D:\users\t-yicxu\data\race\entail_train_concat_options_tri_middle_sorted.tsv --dev_path D:\users\t-yicxu\data\race\entail_dev_concat_options_tri_middle_sorted.tsv --test_path D:\users\t-yicxu\data\race\entail_dev_concat_options_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_tri_middle.txt --suffix race_tri_opt_middle --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 32 --MP_dim 5 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 400 --use_options --wo_sort_instance_based_on_length --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 30 --max_hyp_length 80

