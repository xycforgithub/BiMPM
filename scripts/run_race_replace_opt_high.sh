export TF_CPP_MIN_LOG_LEVEL=1
cd /scratch/BiMPM
nvidia-smi
python src/SentenceMatchTrainer.py --train_path ../data/race/entail_train_replace_options_high_sorted.tsv --dev_path ../data/race/entail_dev_replace_options_high_sorted.tsv --test_path ../data/race/entail_dev_replace_options_high_sorted.tsv --word_vec_path ../data/race/word2vec_entail_train_high.txt --suffix race_replace_opt_high --fix_word_vec --model_dir ../model_data/BiMPM/race --batch_size 32 --MP_dim 10 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 500 --use_options --wo_sort_instance_based_on_length --learning_rate 0.001


