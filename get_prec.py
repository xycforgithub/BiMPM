import numpy as np
import json

entail_score_file=open(r'D:\users\t-yicxu\model_data\BiMPM\SentenceMatch.squad_scratch_2class_nounk_sentOnly.probs',encoding='utf-8')
# gt_file=open(r'D:\users\t-yicxu\data\squad\entail_test_10choice_sent.tsv',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\squad\dev\dev-stanford.json',encoding='utf-8')
recallat=1

all_data={'data':[]}
line=input_data.readline()
# print(line)
assert line.strip()=='SQuDA'
for line in input_data:
	all_data['data'].append(json.loads(line))

correct_count=0
for data in all_data['data']:
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

print('recall@%d:%f' % (recallat, correct_count/len(all_data['data'])))

