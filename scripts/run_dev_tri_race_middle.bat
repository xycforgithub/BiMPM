set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=2
set TF_CPP_MIN_LOG_LEVEL=1
rem python src/TriMatchDecoder.py --in_path D:\users\t-yicxu\data\race\entail_dev_concat_options_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_dev_tri_middle.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_ori_middle_opt4 --out_path D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_middle_opt4_dev_3.probs
python src/TriMatchDecoder.py --in_path D:\users\t-yicxu\data\race\entail_dev_concat_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_middle_dev_tri_new.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_ori_middle_opt3 --out_path D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_middle_opt3_dev_3.probs