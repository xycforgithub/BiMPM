import json
import numpy as np

mode='dev'
n_ans=1
n_sent=1
span_mode='exact'  # exact or overlap
num_class=3
input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'-v1.1.json',encoding='utf-8')
dump_data=open(r'D:\users\t-yicxu\squad_fromxd\squad\squad_data\squda-'+mode+'.dump',encoding='utf-8')
output_data=open(r'D:\users\t-yicxu\data\squad\entail_'+mode+'_%d_%d_%s_%dclass.tsv' %(n_ans,n_sent,span_mode,num_class),encoding='utf-8')

texts=[]
hyps=[]
labels=[]
ids=[]

all_data=json.load(input_data)
for (ii,data) in enumerate(all_data['data']):
	if ii % 1000==0:
		print('data',ii)
	try:
		assert len(data['answer_pos'])==1
		assert len(data['answer_pos'][0])==2
		assert len(data['answer_pos'][0][0])==2
	except AssertionError:
		print(data['answer_pos'])
		input('check')
	gt_sent=range(data['answer_pos'][0][0][0],data['answer_pos'][0][1][0])
	gt_sent_texts=[' '.join([t for t in data['context_tokens'][k]]) for k in gt_sent]
	gt_text=' '.join(gt_sent_texts)
	ques_text=' '.join(data['question_tokens'])
	n_context_token=sum([len(sent) for sent in data['context_tokens']])


	# Insert ground truth
	gt_ans_words=[]
	for sentid in gt_sent:
		st_pos=data['answer_pos'][0][0][1] if sentid==data['answer_pos'][0][0][0] else 0
		end_pos=data['answer_pos'][0][1][1]+1 if sentid==data['answer_pos'][0][1][0] else len(data['context_tokens'][sentid])
		for idx in range(st_pos,end_pos):
			gt_ans_words.append(data['context_tokens'][sentid][idx])
	gt_ans=' '.join(gt_ans_words)
	labels.append(1)
	texts.append(gt_text)
	thishyp=' '.join([ques_text,gt_ans])
	hyps.append(thishyp)
	ids.append(data['id']+'_gt')
	print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
	input('check')

	#insert potential wrong answers
	span_probs=[]
	spans=[]
	dump_line=next(dump_data)
	token_probs=line.split(' ')

	# if i>=len(proc_passages):
	# 	break
	if len(token_probs)!=n_context_token:
		print('prediction length=', len(token_probs))
		print('tokenize length=',n_context_token)
	counter+=1
	startps=[]
	endps=[]
	for (tid,tp) in enumerate(token_probs):
		pid,startp,endp=tp.split('#')
		assert tid==int(pid)
		startps.append(float(startp))
		endps.append(float(endp))
	for id1 in range(n_context_token):
		for id2 in range(i,n_context_token):
			span_probs.append(startps[id1]*endps[id2])
			spans.append((id1,id2))
	span_rank=np.argsort(span_probs)
	partsum=[0]
	all_tokens=sum(data['context_tokens'],[])
	for sid in len(data['context_tokens']):
		partsum.append(partsum[-1]+len(data['context_tokens'][sid]))
	gt_span_start=partsum[data['answer_pos'][0][0][0]]+data['answer_pos'][0][0][1]
	gt_span_end=partsum[data['answer_pos'][0][1][0]]+data['answer_pos'][0][1][1]
	added_count=0
	for i in range(len(spans)):
		this_start=spans[span_rank[i]][0]
		this_end=spans[span_rank[i]][1]
		if span_mode=='exact':
			if this_start==gt_span_start and this_end==gt_span_end:
				continue
		else:
			assert span_mode=='overlap'
			if (this_start - gt_span_end)*(this_end - gt_span_start)<=0:
				continue
		texts.append(gt_text)
		this_ans=' '.join([all_tokens[idx] for idx in range(this_start,this_end+1)])
		thishyp=' '.join([ques_text,this_ans])
		if num_class==2:
			labels.append(0)
		else:
			labels.append(2)
		ids.append(data['id']+'_ans')
		print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
		input('check')
		added_count+=1
		if added_count==n_ans:
			break


	#insert wrong texts
	available_sents=set(range(len(data['context_tokens'])))-set(gt_sent)
	choose_num=min(n_sent,len(available_sents))
	chosen_sents=np.random.choice(available_sents,choose_num)
	for sentid in chosen_sents:
		texts.append(' '.join(data['context_tokens'][sentid]))
		hyps.append(' '.join([ques_text,gt_ans]))
		if num_class==2:
			labels.append(0)
		else:
			labels.append(3)
		ids.append(data['id']+'_sent')
		print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
		input('check')
for i in range(len(hyps)):
	print('%d\t%s\t%s\t%s' % (labels[i], hypstr,textstr,ids[i]),file=output_file)	




