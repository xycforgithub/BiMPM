set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set TF_CPP_MIN_LOG_LEVEL=1
python src/TriMatchDecoder.py --in_path D:\users\t-yicxu\data\race\entail_dev_concat_tri_middle.tsv --word_vec_path D:\users\t-yicxu\data\race\word2vec_entail_dev_middle.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_ori_middle --out_path D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_middle.probs