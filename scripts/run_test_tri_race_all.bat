set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set TF_CPP_MIN_LOG_LEVEL=1
python src/TriMatchDecoder.py --in_path D:\users\t-yicxu\data\race\entail_test_concat_options_tri_all_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_tri_all_alldata.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_ori_all_opt6 --out_path D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_all_opt6_all_test.probs