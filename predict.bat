python data-script\restore.py --span_path D:\users\t-yicxu\data\squad\dev\dev-stanford.spans --raw_path D:\users\t-yicxu\data\squad\dev\dev-stanford.doc --ids_path D:\users\t-yicxu\data\squad\dev\dev-stanford.ids --fout pred.json --seqlen 15 --inpaths D:\users\t-yicxu\BiMPM_1.0\model_data\predict.result
python data-script\evaluate-v1.1.py D:\users\t-yicxu\data\squad\original\dev-v1.1.json pred.json