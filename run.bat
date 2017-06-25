set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=3
python src\SentenceMatchTrainer.py --train_path D:\users\t-yicxu\data\quora\train.tsv --dev_path D:\users\t-yicxu\data\quora\dev.tsv --test_path D:\users\t-yicxu\data\quora\test.tsv --word_vec_path D:\users\t-yicxu\data\quora\wordvec.txt --suffix quora --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\ --batch_size 1 --MP_dim 1