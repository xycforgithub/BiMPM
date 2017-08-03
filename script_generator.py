def generate_script(mode,tied_aggre, tied_match, reasonet_train, keep_first, logit_combine):
	filename='run_race_tri_rl_rn_middle'
	model_name='race_tri_rl_rn'

	if tied_aggre:
		filename+='_tied_aggre'
		model_name+='_tied_aggre'
	else:
		filename+='_nontie_aggre'
		model_name+='_nontie_aggre'
	if tied_match:
		filename+='_tied_match'
		model_name+='_tied_match'
	else:
		filename+='_nontie_match'
		model_name+='_nontie_match'
	if reasonet_train=='original':
		filename+='_rn_ori'
		model_name+='_rn_ori'
	else:		
		filename+='_rn_soft'
		model_name+='_rn_soft'
	if keep_first:
		filename+='_keepfirst'
		model_name+='_keepfirst'
	if logit_combine=='sum':
		filename+='_sumlogit'
		model_name+='_sumlogit'
	else:
		filename+='_maxlogit'
		model_name+='_maxlogit'		
	if mode=='windows':
		line_ending='\r\n'
		filename+='.bat'
	else:
		line_ending='\n'
		filename+='.sh'
	out_file=open(filename,'w',newline=line_ending)
	if mode=='windows':
		print('set CUDA_DEVICE_ORDER=PCI_BUS_ID',file=out_file)
		print('set!!! CUDA_VISIBLE_DEVICES=2',file=out_file)
		print('set TF_CPP_MIN_LOG_LEVEL=2',file=out_file)
		base_string=r'python src\TriMatchTrainer_v2.py --train_path D:\users\t-yicxu\data\race_clean\entail_train_concat_options_tri_middle_sorted.tsv --dev_path D:\users\t-yicxu\data\race_clean\entail_dev_concat_options_tri_middle_sorted.tsv --test_path D:\users\t-yicxu\data\race_clean\entail_test_concat_options_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race_clean\word2vec_entail_tri_middle_alldata.txt --suffix %s --fix_word_vec --model_dir D:\users\t-yicxu\model_data\BiMPM\race --batch_size 64 --MP_dim 10 --max_sent_length 500 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 50 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 7 --predict_val --rl_training_method contrastive_imp --rl_matches [0,1,2] --with_highway --with_match_highway --with_aggregation_highway --concat_context --efficient --reasonet_training --reasonet_steps 5 --cond_training' % model_name
	else:
		print('export CUDA_DEVICE_ORDER=PCI_BUS_ID',file=out_file)
		print('export!!! CUDA_VISIBLE_DEVICES=2',file=out_file)
		print('export TF_CPP_MIN_LOG_LEVEL=2',file=out_file)
		print('export PYTHONUNBUFFERED=x',file=out_file)
		base_string=r'python src/TriMatchTrainer_v2.py --train_path ../data/race/entail_train_concat_options_tri_middle_sorted.tsv --dev_path ../data/race/entail_dev_concat_options_tri_middle_sorted.tsv --test_path ../data/race/entail_test_concat_options_tri_middle_sorted.tsv --word_vec_path ../data/race/word2vec_entail_tri_middle_alldata.txt --suffix %s --fix_word_vec --model_dir ../model_data/BiMPM/race --batch_size 64 --MP_dim 10 --max_sent_length 500 --learning_rate 0.001 --max_epochs 10 --match_to_passage --display_every 50 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 7 --predict_val --rl_training_method contrastive_imp --rl_matches [0,1,2] --with_highway --with_match_highway --with_aggregation_highway --concat_context --efficient --reasonet_training --reasonet_steps 5 --cond_training' % model_name
	if tied_match:
		base_string+=' --tied_match'
	if tied_aggre:
		base_string+=' --tied_aggre'
	base_string+=(' --reasonet_terminate_mode '+reasonet_train)
	base_string+=(' --reasonet_logit_combine '+logit_combine)
	if keep_first:
		base_string+=' --reasonet_keep_first'
	if mode=='linux':
		base_string+=' | tee res_'+model_name+'.txt'
	else:
		base_string+=' > ../model_data/res_'+model_name+'.txt'
	print(base_string,file=out_file)
	return filename
linux_file_list=[]
windows_file_list=[]
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=False, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='sum'))
windows_file_list.append(generate_script(mode='windows',tied_aggre=False, tied_match=False, reasonet_train='softmax', keep_first=False, logit_combine='sum'))
windows_file_list.append(generate_script(mode='windows',tied_aggre=True, tied_match=True, reasonet_train='softmax', keep_first=False, logit_combine='sum'))
windows_file_list.append(generate_script(mode='windows',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=True, logit_combine='sum'))
windows_file_list.append(generate_script(mode='windows',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=True, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=False, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='softmax', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='softmax', keep_first=True, logit_combine='max_pooling'))

windows_file=open('run_windows.bat','w')
for filename in windows_file_list:
	print(filename,file=windows_file)
linux_file=open('run_linux.sh','w',newline='\n')
for filename in linux_file_list:
	print(filename,file=linux_file)