export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TF_CPP_MIN_LOG_LEVEL=2
python src/TriMatchTrainer_v2.py --train_path ../data/race/entail_train_concat_options_tri_middle_sorted_lpart.tsv --dev_path ../data/race/entail_train_concat_options_tri_middle_sorted_lpart.tsv --test_path ../data/race/entail_train_concat_options_tri_middle_sorted_lpart.tsv --word_vec_path ../data/race/word2vec_entail_tri_middle_lpart_alldata.txt --suffix race_tri_rl_debug --fix_word_vec --model_dir ../model_data/BiMPM/race --suffix race_tri_rl_debug --fix_word_vec --batch_size 64 --MP_dim 10 --max_sent_length 500 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 1 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 7 --create_new_model --predict_val --rl_training_method contrastive_imp --rl_matches [0,1,2] --with_highway --with_match_highway --with_aggregation_highway --concat_context --efficient

