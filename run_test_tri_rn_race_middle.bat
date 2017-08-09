set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
set TF_CPP_MIN_LOG_LEVEL=1
python src/TriMatchDecoder_v2.py --in_path D:\users\t-yicxu\data\race_clean\entail_test_concat_options_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race_clean\word2vec_entail_tri_middle_alldata.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_rl_rn_nontie_aggre_nontie_match_rn_ori_sumlogit --out_path D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_rl_rn_nontie_aggre_nontie_match_rn_ori_sumlogit.probs --output_gate_probs