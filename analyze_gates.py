import numpy as np
import random
import json
import scipy.stats
import csv
from collections import defaultdict



test_mode='test'
fileprefix=r'D:\users\t-yicxu\model_data\BiMPM\TriMatch.race_tri_rl_rn_nontie_aggre_nontie_match_rn_ori_sumlogit'
pred_file=open(fileprefix+'.probs')
gate_prob_file=open(fileprefix+'.gateprobs')
out_file=open('../model_data/res_analyze_gates.txt','w',encoding='utf-8')
match_ids=True

input_data=open(r'D:\users\t-yicxu\data\race_clean\entail_%s_concat_options_tri_middle_sorted.tsv' % test_mode,encoding='utf-8')
fieldnames=['label','passage','question','choice','id']
reader=csv.DictReader(input_data,fieldnames=fieldnames,dialect='excel-tab')

num_gates=3
num_steps=5



output_count=0


n_choice=4
label_list=[]
prob_list=[]
pbcount=0
correct_count=0

def process_entail_line(entail_line):
	# print(entail_line)
	choice,entail_probs=entail_line.split('\t')
	label_pbs=[]
	for each_prob in entail_probs.split(' '):
		label,prob = each_prob.split(':')
		label_pbs.append(float(prob))
	prediction=np.argmax(label_pbs)
	return int(choice),label_pbs,prediction
def process_entail_line_nolabel(entail_line):
	label_pbs=[]
	for each_prob in entail_line.split(' '):
		label,prob = each_prob.split(':')
		label_pbs.append(float(prob))
	prediction=np.argmax(label_pbs)
	return label_pbs,prediction

gate_count=np.zeros([3,5],dtype=np.float32)
rn_gate_count=np.zeros([3,5],dtype=np.float32)
sp_gate_count=np.zeros([3,5],dtype=np.float32)
sp_count=0
zcount=0
vocab_count=defaultdict(int)
sp_count2=0
vocab_count_general=defaultdict(int)
vocab_gate_count={}

for lid,entail_line in enumerate(pred_file):

	pbcount+=1
	gt_choice,label_pbs,prediction=process_entail_line(entail_line)
	gate_prob_line=next(gate_prob_file)
	gate_pbs,gate_choice=process_entail_line_nolabel(gate_prob_line)

	choices=[]
	for cid in range(n_choice):
		this_data=next(reader)
		passage=this_data['passage']
		question=this_data['question']
		choices.append(this_data['choice'])
		if cid!=0:
			# print(qid)
			assert qid==this_data['id']
		qid=this_data['id']
		if int(this_data['label'])==1:
			gt_choice=cid
			# print(cid, gt_choice)
			# assert cid==gt_choice
	if gt_choice==prediction:
		correct_count+=1
		print('passage:',passage,file=out_file)
		print('question:',question,file=out_file)
		print('choices:',choices,file=out_file)
		print('gt choice:',gt_choice,'prediction:',prediction,file=out_file)
		rl_gate=gate_choice % num_gates
		rn_step=gate_choice//num_gates
		for word in question.lower().split(' '):
			if word not in vocab_gate_count:
				vocab_gate_count[word]=np.zeros([num_gates,num_steps])
			vocab_gate_count[word][rl_gate,rn_step]+=1
		if 'title' in question.lower():
			sp_gate_count[rl_gate,rn_step]+=1
			sp_count+=1
		if rn_step==0:
			for word in set(question.split(' ')):
				vocab_count[word.lower()]+=1
			sp_count2+=1
		for word in set(question.split(' ')):
			vocab_count_general[word.lower()]+=1
		lower_gate_pbs=np.zeros(num_gates)
		for mid in range(num_gates):
			lower_gate_pbs[mid]=np.sum(gate_pbs[mid::num_gates])
			rn_gate=np.argmax(gate_pbs[mid::num_gates])
			rn_gate_count[mid,rn_gate]+=1
		assert abs(np.sum(lower_gate_pbs)-1)<0.0001
		print('summed rl gate:',np.argmax(lower_gate_pbs),file=out_file)
		print(lower_gate_pbs,file=out_file)
		print('rl_gate:',rl_gate, 'reasonet step:',rn_step,file=out_file)
		print('line:',gate_prob_line,file=out_file)
		gate_count[rl_gate,rn_step]+=1
print('accuracy:',correct_count/pbcount)
print(sp_count)
out_file_tmp='../model_data/temp_res.txt'
for key in vocab_count:
	if vocab_count[key]>2:
		r1=vocab_count[key]/sp_count2*100
		r2=vocab_count_general[key]/correct_count*100
		if r1/r2>1.2:
			print('word ',key,',','{:.1f}%, total:{:.1f}%'.format(r1,r2))
print(sp_gate_count/sp_count)
for word in vocab_gate_count:
	totalnum=np.sum(vocab_gate_count[word])
	if totalnum>5 and vocab_gate_count[word][1,0]/totalnum<0.5:
		print(word)
		print(vocab_gate_count[word]/totalnum)
np.savetxt('../model_data/gate_count.csv',gate_count/np.sum(gate_count)*100,fmt='%.1f',delimiter=',')
np.savetxt('../model_data/rn_gate_count.csv',rn_gate_count/np.sum(gate_count)*100,fmt='%.1f',delimiter=',')
np.savetxt('../model_data/sp_gate_count.csv',sp_gate_count/float(sp_count)*100,fmt='%.1f',delimiter=',')