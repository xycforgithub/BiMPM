export TF_CPP_MIN_LOG_LEVEL=1
cd /scratch/BiMPM
nvidia-smi
python src/TriMatchTrainer.py --train_path ../data/race/entail_train_concat_tri_all_shuffled.tsv --dev_path ../data/race/entail_dev_concat_tri_all.tsv --test_path ../data/race/entail_dev_concat_tri_all.tsv --word_vec_path ../data/race/word2vec_entail_tri_all.txt --suffix race_tri_all_ori_linux --fix_word_vec --model_dir ../model_data/BiMPM/race --batch_size 32 --MP_dim 5 --with_highway --with_match_highway --with_aggregation_highway --max_sent_length 400 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 30 --max_hyp_length 80
