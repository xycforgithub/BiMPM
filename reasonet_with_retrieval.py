import json
import math
import numpy as np
from math import log

n_choice=10
# input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'-v1.1.json',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\squad\dev\dev-stanford.json',encoding='utf-8')
dump_data=open(r'D:\users\t-yicxu\biglearn\res_v16_dev.score.0.dump',encoding='utf-8')
entail_score_file=open(r'D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad_transfer_sentOnly_1.probs')
out_file=open(r'..\model_data\predict.result','w',encoding='utf-8')
best_choice=5 # number or "all"
mode='coeff' # sqrt or direct or coeff
coeff=0.5



# all_data=json.load(input_data)
all_data={'data':[]}
line=input_data.readline()
# print(line)
assert line.strip()=='SQuDA'
for line in input_data:
	all_data['data'].append(json.loads(line))
	# if len(all_data['data']) % 1000==0:
	# 	break

verbose=False
proc_all_data=[]
for (ii,data) in enumerate(all_data['data']):
	if ii % 1000==0:
		print('proc data',ii)
		# break
	
	
	dump_line=next(dump_data)
	ques_text=' '.join(data['question_tokens'])
	n_context_token=sum([len(sent) for sent in data['context_tokens']])
	
	#insert potential wrong answers
	span_probs=[]
	spans=[]



	token_probs=dump_line.split(' ')
	# if i>=len(proc_passages):
	# 	break
	if len(token_probs)!=n_context_token:
		print('question',ii)
		print('prediction length=', len(token_probs))
		print('tokenize length=',n_context_token)
		print(data)
		print(dump_line)
		input('check')
		continue
	startps=[]
	endps=[]
	for (tid,tp) in enumerate(token_probs):
		pid,startp,endp=tp.split('#')
		assert tid==int(pid)
		startps.append(float(startp))
		endps.append(float(endp))
	# print(startps)
	# print(endps)
	# input('check')
	for id1 in range(n_context_token):
		for id2 in range(id1,min(n_context_token,id1+15)):
			span_probs.append(-startps[id1]*endps[id2])
			spans.append((id1,id2))
	span_rank=np.argsort(span_probs)




	n_sent=len(data['context_tokens'])
	sent_score=[]
	for sentid in range(n_sent):
		entail_line=next(entail_score_file)
		choice,entail_probs=entail_line.split('\t')
		find=False
		for each_prob in entail_probs.split(' '):
			label,prob = each_prob.split(':')
			if label=='1':
				this_entail_prob=float(prob)
				find=True
		assert find
		for wid in range(len(data['context_tokens'][sentid])):
			sent_score.append(this_entail_prob)
	# print(len(sent_score))
	# print(len(startps))
	assert len(sent_score)==len(startps)
	if best_choice=='all':
		select_num=len(spans)
	else:
		select_num=best_choice
	best_start=0
	best_end=0
	max_score=-100
	for aid in range(select_num):
		this_start=spans[span_rank[aid]][0]
		this_end=spans[span_rank[aid]][1]
		if mode=='sqrt':
			this_score=log(startps[this_start])+log(endps[this_end])+0.5*log(sent_score[this_start]*sent_score[this_end])
		elif mode=='direct':
			this_score=log(startps[this_start])+log(endps[this_end])+log(sent_score[this_start]*sent_score[this_end])
		else:
			this_score=log(startps[this_start])+log(endps[this_end])+coeff*log(sent_score[this_start]*sent_score[this_end])
		if this_score>max_score:
			best_start=this_start
			best_end=this_end
			max_score=this_score
	print('%d\t%d\t%f\t%f\t%f' % (best_start,best_end,startps[best_start],endps[best_end],max_score), file=out_file)
