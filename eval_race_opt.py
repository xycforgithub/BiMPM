import numpy as np
import random
import json

test_mode='dev'
subset='middle'
pred_file=open(r'd:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_concat_opt_middle_noright2_dev.probs')
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
correctcount=0
buffers=[]
subcount=0
subtotcount=0
pointer=0
for entail_line in pred_file:
	try:
		choice,entail_probs=entail_line.split('\t')
	except ValueError:
		print(entail_line)
		input('check')
	# print(entail_line)
	# entail_probs=entail_line
	label_pbs=[]
	for each_prob in entail_probs.split(' '):
		label,prob = each_prob.split(':')
		label_pbs.append(float(prob))
	prediction=np.argmax(label_pbs)
	this_counter=0
	for t in label_pbs:
		if t==label_pbs[prediction]:
			this_counter+=1
	if this_counter!=1:
		print('same: amount', this_counter)
		print(entail_line)
		input('check')
	if prediction==int(choice):
		correctcount+=1
		subcount+=1
	pbcount+=1
	subtotcount+=1
	buffers.append(entail_line)
	if subtotcount==totals[pointer]/4:
		subtotcount=0
		# if corrects[pointer]!=subcount:
		# 	for string in buffers:
		# 		print(string)
		# 	print(subcount)
		# 	print(corrects[pointer])
		buffers=[]
		subcount=0
		pointer+=1




print('accuracy: ',correctcount/pbcount*100)
