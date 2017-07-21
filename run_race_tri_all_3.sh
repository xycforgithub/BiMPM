export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=x
python src/TriMatchTrainer.py --train_path ../data/race/entail_train_concat_options_tri_all_sorted.tsv --dev_path ../data/race/entail_dev_concat_options_tri_all_sorted.tsv --test_path ../data/race/entail_test_concat_options_tri_all_sorted.tsv --word_vec_path ../data/race/word2vec_entail_all_tri_alldata.txt --suffix race_tri_ori_all_opt3 --fix_word_vec --model_dir ../model_data/BiMPM/race --batch_size 32 --MP_dim 10 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 500 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 30 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 3 --predict_val



