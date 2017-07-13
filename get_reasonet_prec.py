import numpy as np
import json
from tqdm import tqdm

entail_score_file=open(r'D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad_transfer_sentOnly_2.probs',encoding='utf-8')
# gt_file=open(r'D:\users\t-yicxu\data\squad\entail_test_10choice_sent.tsv',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\squad\dev\dev-stanford.json',encoding='utf-8')
dump_data=open(r'D:\users\t-yicxu\biglearn\res_v16_dev.score.0.dump',encoding='utf-8')
recallat=1
either_correct_count=0

reasonet_best_choice=1

all_data={'data':[]}
line=input_data.readline()
# print(line)
assert line.strip()=='SQuDA'
for line in input_data:
	all_data['data'].append(json.loads(line))

correct_count=0
reasonet_correct_count=0
reasonet_with_bimpm=0
for data in tqdm(all_data['data']):
	n_sent=len(data['context_tokens'])
	probs=[]
	labels=[]
	for sentid in range(n_sent):
		entail_line=next(entail_score_file)
		choice,entail_probs=entail_line.split('\t')
		labels.append(int(choice))
		find=False
		for each_prob in entail_probs.split(' '):
			label,prob = each_prob.split(':')
			if label=='1':
				this_entail_prob=float(prob)
				find=True
		assert find
		probs.append(-this_entail_prob)
	ranks=np.argsort(probs)
	for sentid in range(0,recallat):
		if labels[ranks[sentid]]==1:
			correct_count+=1
			break
	dump_line=next(dump_data)
	span_probs=[]
	spans=[]


	n_context_token=sum([len(sent) for sent in data['context_tokens']])
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

	partsum=[0]
	all_tokens=sum(data['context_tokens'],[])
	for sid in range(len(data['context_tokens'])):
		partsum.append(partsum[-1]+len(data['context_tokens'][sid]))
	for aid in range(reasonet_best_choice):
		for sid in range(n_sent):
			if partsum[sid+1]>spans[span_rank[aid]][0]:
				break
		this_sid=sid
		if labels[this_sid]==1:
			reasonet_correct_count+=1
		if this_sid==ranks[0]:
			reasonet_with_bimpm+=1
		if labels[this_sid]==1 or labels[ranks[0]]==1:
			either_correct_count+=1





print('prec@%d:%f' % (recallat, correct_count/len(all_data['data'])))
print('reasonet prec@%d:%f' % (reasonet_best_choice, reasonet_correct_count/len(all_data['data'])))
print('collapse:%f' % (reasonet_with_bimpm/len(all_data['data'])))
print('either prec@2:%f' % (either_correct_count/len(all_data['data'])))


