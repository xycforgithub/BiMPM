import numpy as np
import random
import json

test_mode='test'
subset='all'
pred_file=open(r'd:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_concat_all_test.probs')
input_data2=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_high.json',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_middle.json',encoding='utf-8')
if subset=='high':
	all_data=json.load(input_data2)
elif subset=='middle':
	all_data=json.load(input_data)
else:
	all_data=json.load(input_data)
	data2=json.load(input_data2)
	all_data['data']+=data2['data']

out_file=open(r'../model_data/result.txt','w',encoding='utf-8')

n_choice=4
label_list=[]
prob_list=[]
pbcount=0
correctcount=0
for entail_line in pred_file:
	try:
		choice,entail_probs=entail_line.split('\t')
	except ValueError:
		print(entail_line)
		input('check')
	# print(entail_line)
	# entail_probs=entail_line
	find=False
	for each_prob in entail_probs.split(' '):
		label,prob = each_prob.split(':')
		if label=='1':
			this_entail_prob=float(prob)
			find=True
	assert find
	label_list.append(int(choice))
	prob_list.append(this_entail_prob)
	if len(label_list)==n_choice:
		pred_ans=np.argmax(prob_list)
		gt_ans=np.argmax(label_list)
		assert(int(label_list[gt_ans])==1)
		if pred_ans==gt_ans:
			correctcount+=1
		if random.random()>0.5 and pred_ans==gt_ans :
			data=all_data['data'][pbcount]
			print('passage=',' '.join(data['document']),'question=',' '.join(data['question']),
				'options=',data['options'],'pred=',pred_ans,'gt=',gt_ans,'pred_gap=',prob_list[gt_ans]-np.mean(prob_list),file=out_file)
		pbcount+=1
		# print(pred_ans,gt_ans)
		# input('check')
		label_list=[]
		prob_list=[]
print('accuracy:',correctcount/pbcount)
