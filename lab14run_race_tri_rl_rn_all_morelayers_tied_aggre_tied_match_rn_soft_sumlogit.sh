export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=x
python src/TriMatchTrainer_v2.py --train_path ../data/race/entail_train_concat_options_tri_all_sorted.tsv --dev_path ../data/race/entail_dev_concat_options_tri_all_sorted.tsv --test_path ../data/race/entail_test_concat_options_tri_all_sorted.tsv --word_vec_path ../data/race/word2vec_entail_tri_all_alldata.txt --model_dir ../model_data/BiMPM/race --suffix race_tri_rl_rn_all_morelayers_tied_aggre_tied_match_rn_soft_sumlogit --fix_word_vec --batch_size 32 --MP_dim 10 --max_sent_length 500 --learning_rate 0.001 --max_epochs 20 --match_to_passage --display_every 50 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 7 --predict_val --rl_training_method contrastive_imp --with_highway --with_match_highway --with_aggregation_highway --concat_context --efficient --reasonet_training --reasonet_steps 5 --cond_training --dropout_rate 0.1 --rl_matches [0,1,2]  --tied_match --tied_aggre --reasonet_terminate_mode softmax --reasonet_logit_combine sum --context_layer_num 2 --aggregation_layer_num 2 | tee res_race_tri_rl_rn_all_morelayers_tied_aggre_tied_match_rn_soft_sumlogit.txt
