import json
import numpy as np

n_choice=10
predict=False
mode='normal' # cheat or normal or sent
# input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'-v1.1.json',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\squad\dev\dev-stanford.json',encoding='utf-8')
dump_data=open(r'D:\users\t-yicxu\biglearn\res_v16_dev.score.0.dump',encoding='utf-8')
if mode=='normal':
	output_file=open(r'D:\users\t-yicxu\data\squad\entail_test_%dchoice.tsv' %(n_choice),'w',encoding='utf-8')
elif mode=='cheat':
	output_file=open(r'D:\users\t-yicxu\data\squad\entail_test_%dchoice_cheat.tsv' %(n_choice),'w',encoding='utf-8')
elif mode=='sent':
	output_file=open(r'D:\users\t-yicxu\data\squad\entail_test_%dchoice_sent.tsv' %(n_choice),'w',encoding='utf-8')


texts=[]
hyps=[]
labels=[]
ids=[]
prediction={}

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
	answer_datas=data['answer_pos']
	ans_data_collection=set()
	for a_data in answer_datas:
		ans_data_collection.add(((a_data[0][0],a_data[0][1]),(a_data[1][0],a_data[1][1])))
	new_answers=[]
	for tup in ans_data_collection:
		new_answers.append([[tup[0][0],tup[0][1]],[tup[1][0],tup[1][1]]])
	if verbose:
		print('new_answers=',new_answers)

	gt_sents=set()
	for ans_span in new_answers:

		gt_sent=range(ans_span[0][0],ans_span[1][0]+1)
		gt_sent_texts=[' '.join([t for t in data['context_tokens'][k]]) for k in gt_sent]
		gt_text=' '.join(gt_sent_texts)
		for sentid in gt_sent:
			gt_sents.add(sentid)


	if mode=='sent':
		for (sentid,sent_data) in enumerate(data['context_tokens']):
			this_sent=' '.join(sent_data)
			hyps.append(ques_text)
			texts.append(this_sent)
			ids.append(data['id']+'sent_%d' % sentid)
			if sentid in gt_sents:
				labels.append(1)
			else:
				labels.append(0)
			if verbose:
				thisl='label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1])
				print(thisl.encode('utf-8'))
				input('check')
			
	else:
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
		added_count=0

		for i in range(n_choice):
			this_start=spans[span_rank[i]][0]
			this_end=spans[span_rank[i]][1]

			this_ans=' '.join([all_tokens[idx] for idx in range(this_start,this_end+1)])
			if i==0:
				prediction[data['id']]=this_ans
			if verbose:
				print('this_start=',this_start,'this_end=',this_end)
				# print('gt_start=',gt_span_start,'gt_end=',gt_span_end)
			for start_sent in range(len(partsum)):
				if partsum[start_sent+1]>=this_start:
					break
			for end_sent in range(len(partsum)):
				if partsum[end_sent]>this_end:
					break
			if mode=='normal':
				text_sent=range(start_sent,end_sent)
				this_text=' '.join([' '.join([t for t in data['context_tokens'][k]]) for k in text_sent])

				texts.append(this_text)
			else:
				texts.append(gt_text)
			this_ans=' '.join([all_tokens[idx] for idx in range(this_start,this_end+1)])
			thishyp=' '.join([ques_text,this_ans])
			hyps.append(thishyp)
			labels.append(1)
			ids.append(data['id']+'_ans')
			if verbose:
				thisl='label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1])
				print(thisl.encode('utf-8'))
				input('check')




		# print('inside here')
		# break
if predict:
	predict_out=open(r'D:\users\t-yicxu\BiMPM_1.0\model_data\sample_predict_tmp.json','w',encoding='utf-8')
	json.dump(prediction,predict_out,ensure_ascii=False)
for i in range(len(hyps)):
	print('%d\t%s\t%s\t%s' % (labels[i], texts[i],hyps[i],ids[i]),file=output_file)	
text_lens=[len(a) for a in texts]
hyp_lens=[len(a) for a in hyps]

n_words_text=[]
n_words_hyp=[]
for i in range(len(hyps)):
	n_words=len(hyps[i].split(' '))
	n_words_hyp.append(n_words)

	n_words=len(texts[i].split(' '))
	n_words_text.append(n_words)


print('mean text length:',np.mean(text_lens))
print('mean hyp length:',np.mean(hyp_lens))

print('max text length:',np.max(text_lens))
print('max hyp length:',np.max(hyp_lens))

print('mean text length:',np.mean(n_words_text))
print('mean hyp length:',np.mean(n_words_hyp))

print('max text length:',np.max(n_words_text))
print('max hyp length:',np.max(n_words_hyp))








	