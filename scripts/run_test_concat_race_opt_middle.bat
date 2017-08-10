set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=3
set TF_CPP_MIN_LOG_LEVEL=1
python src/SentenceMatchDecoder.py --in_path D:\users\t-yicxu\data\race_clean\entail_test_concat_options_all_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race_clean\word2vec_entail_all_alldata.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\SentenceMatch.race_concat_opt_all_noright --out_path D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_concat_opt_all_noright_test_all.probs --use_options