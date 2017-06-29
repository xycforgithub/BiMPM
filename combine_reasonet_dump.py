import json
import numpy as np
from math import log
import sys

dump_file=open(r'D:\users\t-yicxu\biglearn\res_v16_dev.score.0.dump',encoding='utf-8')
entail_score_file=open(r'D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad.test.squad_transfer_2class_randomunk.probs')
out_file=open(r'..\model_data\predict.result','w',encoding='utf-8')
n_class=2
n_choice=10
if len(sys.argv)>=2:
	mix_rate=float(sys.argv[1])
else:
	mix_rate=1.0

for dump_line in dump_file:
	span_probs=[]
	spans=[]
	
	token_probs=dump_line.split(' ')


	startps=[]
	endps=[]
	for (tid,tp) in enumerate(token_probs):
		pid,startp,endp=tp.split('#')
		assert tid==int(pid)
		startps.append(float(startp))
		endps.append(float(endp))
	n_context_token=len(startps)
	assert len(endps)==n_context_token
	# print(startps)
	# print(endps)
	# input('check')
	for id1 in range(n_context_token):
		for id2 in range(id1,min(n_context_token,id1+15)):
			span_probs.append(-startps[id1]*endps[id2])
			spans.append((id1,id2))
	span_rank=np.argsort(span_probs)
	all_scores=[]
	for i in range(n_choice):
		entail_line=next(entail_score_file)
		choice,entail_probs=entail_line.split('\t')
		find=False
		for each_prob in entail_probs.split(' '):
			label,prob = each_prob.split(':')
			if label=='1':
				this_entail_prob=float(prob)
				find=True
		assert find
		all_scores.append(log(-span_probs[span_rank[i]])+mix_rate*log(this_entail_prob))
	maxid=np.argmax(all_scores)
	# print('max_span=', spans[span_rank[0]][0],spans[span_rank[0]][1],'prob=',span_probs[span_rank[0]])
	# print(all_scores)
	# print(maxid)
	# print(span_rank[maxid])
	# print(len(spans))
	# input('check')
	span_start=spans[span_rank[maxid]][0]
	span_end=spans[span_rank[maxid]][1]
	print('%d\t%d\t%f\t%f\t%f' % (span_start,span_end,startps[span_start],endps[span_end],all_scores[maxid]), file=out_file)


	
