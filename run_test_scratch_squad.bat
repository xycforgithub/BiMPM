set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=3
set TF_CPP_MIN_LOG_LEVEL=1
rem python src/SentenceMatchDecoder.py --in_path D:\users\t-yicxu\data\snli_1.0\test.tsv --word_vec_path D:\users\t-yicxu\data\snli_1.0\word2vec.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.snli --out_path D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.snli.test.probs
rem python src/SentenceMatchDecoder.py --in_path D:\users\t-yicxu\data\squad\entail_dev_same_1_overlap_2class.tsv --word_vec_path D:\users\t-yicxu\data\snli_1.0\word2vec.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.snli_new --out_path D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad.test.snli_2class_randomunk.probs
python src/SentenceMatchDecoder.py --in_path D:\users\t-yicxu\data\squad\entail_test_10choice.tsv --word_vec_path D:\users\t-yicxu\data\squad\word2vec_nounk.txt --mode probs --model_prefix D:\users\t-yicxu\model_data\BiMPM\squad\SentenceMatch.squad_scratch_2class_randomunk --out_path D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad.test.squad_scratch_2class_nounk.probs