import json
import numpy as np

mode='train'
n_ans='same'
n_sent=0
span_mode='f1'  # exact or overlap or f1
num_class=2
predict=False
verbose=False
# input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'-v1.1.json',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'\\'+mode+'-stanford.json',encoding='utf-8')
if mode=='dev':
	dump_data=open(r'D:\users\t-yicxu\biglearn\res_v16_dev.score.0.dump',encoding='utf-8')
else:
	dump_data=open(r'D:\users\t-yicxu\biglearn\res_v16_train.score.0.dump',encoding='utf-8')
output_file=open(r'D:\users\t-yicxu\data\squad\entail_'+mode+'_%s_%d_%s_%dclass.tsv' %(str(n_ans),n_sent,span_mode,num_class),'w',encoding='utf-8')

	

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


proc_all_data=[]
for (ii,data) in enumerate(all_data['data']):
	if ii % 1000==0:
		print('proc data',ii)
	try:
		# assert len(data['answer_pos'])==1
		assert len(data['answer_pos'][0])==2
		assert len(data['answer_pos'][0][0])==2
	except AssertionError:
		print(data['answer_pos'])
		input('check')
	answer_datas=data['answer_pos']
	ans_data_collection=set()
	for a_data in answer_datas:
		ans_data_collection.add(((a_data[0][0],a_data[0][1]),(a_data[1][0],a_data[1][1])))
	new_answers=[]
	for tup in ans_data_collection:
		new_answers.append([[tup[0][0],tup[0][1]],[tup[1][0],tup[1][1]]])
	if verbose:
		print('new_answers=',new_answers)
	dump_line=next(dump_data)
	for ans_span in new_answers:

		gt_sent=range(ans_span[0][0],ans_span[1][0]+1)
		gt_sent_texts=[' '.join([t for t in data['context_tokens'][k]]) for k in gt_sent]
		gt_text=' '.join(gt_sent_texts)
		ques_text=' '.join(data['question_tokens'])
		n_context_token=sum([len(sent) for sent in data['context_tokens']])


		# Insert ground truth
		gt_ans_words=[]
		for sentid in gt_sent:
			st_pos=ans_span[0][1] if sentid==ans_span[0][0] else 0
			end_pos=ans_span[1][1] if sentid==ans_span[1][0] else len(data['context_tokens'][sentid])-1
			# print(st_pos,end_pos)
			for idx in range(st_pos,end_pos+1):
				gt_ans_words.append(data['context_tokens'][sentid][idx])
		gt_ans=' '.join(gt_ans_words)
		labels.append(1)
		texts.append(gt_text)
		thishyp=' '.join([ques_text,gt_ans])
		hyps.append(thishyp)
		ids.append(data['id']+'_gt')
		if verbose:
			print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
			input('check')

		#insert wrong texts
		available_sents=set(range(len(data['context_tokens'])))-set(gt_sent)
		choose_num=min(n_sent,len(available_sents))
		if choose_num==0:
			continue
		chosen_sents=np.random.choice(list(available_sents),choose_num)
		for sentid in chosen_sents:
			texts.append(' '.join(data['context_tokens'][sentid]))
			hyps.append(' '.join([ques_text,gt_ans]))
			if num_class==2:
				labels.append(0)
			else:
				labels.append(3)
			ids.append(data['id']+'_sent')
			if verbose:
				print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
				input('check')			

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
	partsum=[0]
	all_tokens=sum(data['context_tokens'],[])
	for sid in range(len(data['context_tokens'])):
		partsum.append(partsum[-1]+len(data['context_tokens'][sid]))

	added_count=0
	if n_ans=='same':
		target_num=len(new_answers)
	else:
		target_num=n_ans
	first_ten_perm=np.random.permutation(10)
	for tmpi in range(len(spans)):
		if tmpi<10:
			i=first_ten_perm[tmpi]
		else:
			i=tmpi
		this_start=spans[span_rank[i]][0]
		this_end=spans[span_rank[i]][1]

		this_ans=' '.join([all_tokens[idx] for idx in range(this_start,this_end+1)])
		if i==0:
			prediction[data['id']]=this_ans
		if verbose:
			print('this_start=',this_start,'this_end=',this_end)
			# print('gt_start=',gt_span_start,'gt_end=',gt_span_end)

		canuse=True
		for a_gt_span in new_answers:
			a_gt_span_start=partsum[a_gt_span[0][0]]+a_gt_span[0][1]
			a_gt_span_end=partsum[a_gt_span[1][0]]+a_gt_span[1][1]
			if span_mode=='exact':
				if this_start==a_gt_span_start and this_end==a_gt_span_end:
					canuse=False
			elif span_mode=='f1':
				overlap_length=max(min(a_gt_span_end,this_end)+1-max(a_gt_span_start,this_start),0)
				if overlap_length==0:
					continue
				else:
					precision=overlap_length/(this_end-this_start+1)
					recall=overlap_length/(a_gt_span_end - a_gt_span_start+1)
					f1=2*precision*recall/(precision+recall)
					if f1>=0.5:
						canuse=False
						break
			else:
				assert span_mode=='overlap'
				if (this_start - a_gt_span_end)*(this_end - a_gt_span_start)<=0:
					canuse=False


		if not canuse:
			continue
		for start_sent in range(len(partsum)):
			if partsum[start_sent+1]>=this_start:
				break
		for end_sent in range(len(partsum)):
			if partsum[end_sent]>=this_end:
				break

		text_sent=range(start_sent,end_sent)
		this_text=' '.join([' '.join([t for t in data['context_tokens'][k]]) for k in text_sent])

		texts.append(this_text)
		this_ans=' '.join([all_tokens[idx] for idx in range(this_start,this_end+1)])
		thishyp=' '.join([ques_text,this_ans])
		hyps.append(thishyp)
		if num_class==2:
			labels.append(0)
		else:
			labels.append(2)
		ids.append(data['id']+'_ans_%d' % (tmpi))
		if verbose:
			print('label=%d, text=%s, hyp=%s, id=%s' % (labels[-1],texts[-1],hyps[-1],ids[-1]))
			input('check')
		added_count+=1
		if added_count==target_num:
			break



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






