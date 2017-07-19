set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2
set TF_CPP_MIN_LOG_LEVEL=1
python src/SentenceMatchDecoder.py --in_path D:\users\t-yicxu\data\race\entail_test_concat_options_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_test_middle.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\SentenceMatch.race_concat_opt_middle_noright --out_path D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_concat_opt_middle_noright_test.probs --use_options