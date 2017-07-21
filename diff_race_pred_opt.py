import numpy as np
import random
import json
import scipy.stats

test_mode='test'
subset='all'
pred_file1=open(r'd:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_middle_opt4_test.probs')
pred_file2=open(r'd:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_ori_middle_opt6_test.probs')
# input_data2=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_high.json',encoding='utf-8')
# input_data=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_middle.json',encoding='utf-8')
# if subset=='high':
# 	all_data=json.load(input_data2)
# elif subset=='middle':
# 	all_data=json.load(input_data)
# else:
# 	all_data=json.load(input_data)
# 	data2=json.load(input_data2)
# 	all_data['data']+=data2['data']

# out_file=open(r'../model_data/result.txt','w',encoding='utf-8')
data=json.load(open('res.txt'))
corrects=data['correct']
totals=data['total']

n_choice=4
label_list=[]
prob_list=[]
pbcount=0
correctcount1=0
correctcount2=0
either_correct_count=0
diffcount=0
diffcount_raw=0
vote_correct_count=0
buffers=[]
subcount=0
subtotcount=0
pointer=0
sum_kl=0.0

for entail_line1 in pred_file1:
	def process_entail_line(entail_line):
		choice,entail_probs=entail_line.split('\t')
		label_pbs=[]
		for each_prob in entail_probs.split(' '):
			label,prob = each_prob.split(':')
			label_pbs.append(float(prob))
		prediction=np.argmax(label_pbs)
		return int(choice),label_pbs,prediction

	choice1,label_pbs1,prediction1=process_entail_line(entail_line1)
	entail_line2=next(pred_file2)	
	choice2,label_pbs2,prediction2=process_entail_line(entail_line2)
	assert choice1==choice2

	pbcount+=1
	if prediction1==choice1:
		correctcount1+=1
	if prediction2==choice1:
		correctcount2+=1
	if prediction2==choice1 and prediction1==choice1:
		either_correct_count+=1
	if prediction1!=prediction2:
		diffcount_raw+=1
	if prediction1!=prediction2 and (prediction1==choice1 or prediction2==choice2):
		diffcount+=1
	vote_pbs=[label_pbs1[i]+label_pbs2[i] for i in range(len(label_pbs1))]
	vote_pred=np.argmax(vote_pbs)
	if vote_pred==choice1:
		vote_correct_count+=1
	sum_kl+=scipy.stats.entropy(label_pbs1,label_pbs2)





print('accuracy1: ',correctcount1/pbcount*100)
print('accuracy2: ',correctcount2/pbcount*100)
print('Either correct accuracy: ',either_correct_count/pbcount*100)
print('raw diff ratio: ',diffcount_raw/pbcount*100)
print('meaningful diff ratio: ',diffcount/pbcount*100)
print('vote accuracy: ',vote_correct_count/pbcount*100)
print('average KL divergence:', sum_kl/pbcount)
