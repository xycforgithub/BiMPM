def generate_script(mode,logit_combine='sum', dataset='middle', increase_layers=False, dropout_rate=0.1, learning_rate=0.001,no_gated_rl=False,tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=False, prefix='',reasonet_config=None ):
	if reasonet_config is not None:
		if reasonet_config in [0,1]:
			tied_aggre=False
			tied_match=False
			reasonet_train='original'
			logit_combine = 'sum' if reasonet_config==0 else 'max_pooling'
		elif reasonet_config==2:
			tied_match=True
			tied_aggre=True
			reasonet_train='softmax'
			logit_combine='sum'
	filename=prefix+'run_race_tri_rl_rn_'+dataset
	model_name='race_tri_rl_rn_'+dataset
	if no_gated_rl:
		filename+='_nogates'
		model_name+='_nogates'
	if increase_layers:
		filename+='_morelayers'
		model_name+='morelayers'
	if dropout_rate!=0.1:
		filename+='_drop'+str(dropout_rate)
		model_name+='_drop'+str(dropout_rate)
	if learning_rate!=0.001:
		filename+='_lr'+str(learning_rate)
		model_name+='_lr'+str(learning_rate)

	if tied_aggre:
		filename+='_tied_aggre'
		model_name+='_tied_aggre'
	if tied_match:
		filename+='_tied_match'
		model_name+='_tied_match'
	if reasonet_train=='softmax':
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
		print('set PYTHONUNBUFFERED=x', file=out_file)
		if dataset=='middle':
			base_string=r'python src\TriMatchTrainer_v2.py --train_path D:\users\t-yicxu\data\race_clean\entail_train_concat_options_tri_middle_sorted.tsv --dev_path D:\users\t-yicxu\data\race_clean\entail_dev_concat_options_tri_middle_sorted.tsv --test_path D:\users\t-yicxu\data\race_clean\entail_test_concat_options_tri_middle_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race_clean\word2vec_entail_tri_middle_alldata.txt '
		else:
			base_string = r'python src\TriMatchTrainer_v2.py --train_path D:\users\t-yicxu\data\race_clean\entail_train_concat_options_tri_all_sorted.tsv --dev_path D:\users\t-yicxu\data\race_clean\entail_dev_concat_options_tri_all_sorted.tsv --test_path D:\users\t-yicxu\data\race_clean\entail_test_concat_options_tri_all_sorted.tsv --word_vec_path D:\users\t-yicxu\data\race_clean\word2vec_entail_tri_all_alldata.txt'

		base_string+=r' --model_dir D:\users\t-yicxu\model_data\BiMPM\race'

	else:
		print('export CUDA_DEVICE_ORDER=PCI_BUS_ID',file=out_file)
		print('export!!! CUDA_VISIBLE_DEVICES=2',file=out_file)
		print('export TF_CPP_MIN_LOG_LEVEL=2',file=out_file)
		print('export PYTHONUNBUFFERED=x',file=out_file)
		if dataset=='middle':
			base_string=r'python src/TriMatchTrainer_v2.py --train_path ../data/race/entail_train_concat_options_tri_middle_sorted.tsv --dev_path ../data/race/entail_dev_concat_options_tri_middle_sorted.tsv --test_path ../data/race/entail_test_concat_options_tri_middle_sorted.tsv --word_vec_path ../data/race/word2vec_entail_tri_middle_alldata.txt'
		else:
			base_string = r'python src/TriMatchTrainer_v2.py --train_path ../data/race/entail_train_concat_options_tri_all_sorted.tsv --dev_path ../data/race/entail_dev_concat_options_tri_all_sorted.tsv --test_path ../data/race/entail_test_concat_options_tri_all_sorted.tsv --word_vec_path ../data/race/word2vec_entail_tri_all_alldata.txt'

		base_string+=r' --model_dir ../model_data/BiMPM/race'

	base_string += r' --suffix %s --fix_word_vec --batch_size 64 --MP_dim 10 --max_sent_length 500 --learning_rate %.3f --max_epochs 20 --match_to_passage --display_every 50 --max_hyp_length 100 --use_options --wo_sort_instance_based_on_length --matching_option 7 --predict_val --rl_training_method contrastive_imp --with_highway --with_match_highway --with_aggregation_highway --concat_context --efficient --reasonet_training --reasonet_steps 5 --cond_training --dropout_rate %.1f' % (model_name, learning_rate, dropout_rate)
	if no_gated_rl:
		base_string+=' --rl_matches [0]'
	else:
		base_string+=' --rl_matches [0,1,2] '
	if tied_match:
		base_string+=' --tied_match'
	if tied_aggre:
		base_string+=' --tied_aggre'
	base_string+=(' --reasonet_terminate_mode '+reasonet_train)
	base_string+=(' --reasonet_logit_combine '+logit_combine)
	if keep_first:
		base_string+=' --reasonet_keep_first'
	if increase_layers:
		base_string+=' --context_layer_num 2 --aggregation_layer_num 2'
	if mode=='linux':
		base_string+=' | tee res_'+model_name+'.txt'
	else:
		base_string+=' > ../model_data/res_'+model_name+'.txt'
	print(base_string,file=out_file)
	return filename
# linux_file_list=[]
windows_file_list1=[]# for 05
windows_file_list2=[]# for 09
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=False, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='sum'))
# windows_file_list.append(generate_script(mode='windows',tied_aggre=False, tied_match=False, reasonet_train='softmax', keep_first=False, logit_combine='sum'))
# windows_file_list.append(generate_script(mode='windows',tied_aggre=True, tied_match=True, reasonet_train='softmax', keep_first=False, logit_combine='sum'))
# windows_file_list.append(generate_script(mode='windows',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=True, logit_combine='sum'))
# windows_file_list.append(generate_script(mode='windows',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=True, logit_combine='sum'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=False, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=False, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='original', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='original', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=False, tied_match=False, reasonet_train='softmax', keep_first=True, logit_combine='max_pooling'))
# linux_file_list.append(generate_script(mode='linux',tied_aggre=True, tied_match=True, reasonet_train='softmax', keep_first=True, logit_combine='max_pooling'))
# windows_file_list1.append(generate_script(mode='windows',logit_combine='sum',dataset='all',increase_layers=False,dropout_rate=0.1, learning_rate=0.001, no_gated_rl=False))
# windows_file_list1.append(generate_script(mode='windows',logit_combine='sum',dataset='all',increase_layers=False,dropout_rate=0.5, learning_rate=0.005, no_gated_rl=False))
#
# generate_script(mode='linux',logit_combine='sum',dataset='middle',increase_layers=False,dropout_rate=0.1, learning_rate=0.001, no_gated_rl=True, prefix='azure5')
# generate_script(mode='linux',logit_combine='sum',dataset='middle',increase_layers=False,dropout_rate=0.5, learning_rate=0.001, no_gated_rl=False, prefix='azure6')
# generate_script(mode='linux',logit_combine='sum',dataset='middle',increase_layers=False,dropout_rate=0.5, learning_rate=0.005, no_gated_rl=False, prefix='lab6')

windows_file_list1.append(generate_script(mode='windows',dataset='middle',increase_layers=True,dropout_rate=0.1, reasonet_config=0))
windows_file_list1.append(generate_script(mode='windows',dataset='middle',increase_layers=True,dropout_rate=0.1, reasonet_config=2))
windows_file_list2.append(generate_script(mode='windows',dataset='all',increase_layers=False,dropout_rate=0.1, reasonet_config=1))
windows_file_list2.append(generate_script(mode='windows',dataset='all',increase_layers=False,dropout_rate=0.1, reasonet_config=2))
windows_file_list2.append(generate_script(mode='windows',dataset='all',increase_layers=False,dropout_rate=0.5, reasonet_config=0))
windows_file_list2.append(generate_script(mode='windows',dataset='all',increase_layers=True,dropout_rate=0.5, reasonet_config=0))
generate_script(mode='linux',dataset='middle',increase_layers=True,dropout_rate=0.5, reasonet_config=0,prefix='azure7')
generate_script(mode='linux',dataset='middle',increase_layers=True,dropout_rate=0.5, reasonet_config=2,prefix='azure8')
generate_script(mode='linux',dataset='middle',increase_layers=True,dropout_rate=0.1, reasonet_config=1,prefix='dev1')
generate_script(mode='linux',dataset='all',increase_layers=True,dropout_rate=0.1, reasonet_config=0,prefix='lab7')
generate_script(mode='linux',dataset='all',increase_layers=True,dropout_rate=0.1, reasonet_config=1,prefix='lab8')
generate_script(mode='linux',dataset='all',increase_layers=True,dropout_rate=0.1, reasonet_config=2,prefix='lab9')




windows_file=open('run_windows_05.bat','w')
for filename in windows_file_list1:
	print('start '+filename,file=windows_file)

windows_file = open('run_windows_09.bat', 'w')
for filename in windows_file_list2:
	print('start '+filename, file=windows_file)
# linux_file=open('run_linux.sh','w',newline='\n')
# for filename in linux_file_list:
# 	print(filename,file=linux_file)