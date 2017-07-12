import json
import numpy as np

mode='dev'
# n_ans='same'
concat_mode='replace' # replace or concat
shuffle=False
shuffle_questions=False
choice_num=4
verbose=False
true_repeat=1
middle_only=False
high_only=True
# input_data=open(r'D:\users\t-yicxu\data\squad\\'+mode+'-v1.1.json',encoding='utf-8')
input_data2=open(r'D:\users\t-yicxu\data\race\processed\\'+mode+'_high.json',encoding='utf-8')
input_data=open(r'D:\users\t-yicxu\data\race\processed\\'+mode+'_middle.json',encoding='utf-8')

if shuffle_questions:
	output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_options_all.tsv' %(concat_mode),'w',encoding='utf-8')
elif middle_only:
	if shuffle:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_middle_shuffled.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')
	else:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_middle.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')
elif high_only:
	if shuffle:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_high_shuffled.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')
	else:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_high.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')

else:
	if shuffle:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_all_shuffled.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')
	else:
		output_file=open(r'D:\users\t-yicxu\data\race\entail_'+mode+'_%s_%d_all.tsv' %(concat_mode,true_repeat),'w',encoding='utf-8')

texts=[]
hyps=[]
labels=[]
ids=[]
prediction={}

if middle_only:
	all_data=json.load(input_data)
	print(len(all_data['data']))
elif high_only:
	all_data=json.load(input_data2)
	print(len(all_data['data']))
else:
	all_data=json.load(input_data)
	all_data2=json.load(input_data2)
	print(len(all_data['data']),len(all_data2['data']))

	all_data['data']=all_data['data']+all_data2['data']

# all_data={'data':[]}
# line=input_data.readline()
# print(line)
# assert line.strip()=='SQuDA'
# for line in input_data:
# 	all_data['data'].append(json.loads(line))

def token_to_text(tokens):
	toks=[]
	for tok in tokens:
		if tok.strip()!='':
			toks.append(tok)
	return ' '.join(toks)

concat_count=0
if shuffle_questions:
	question_order=np.random.permutation(len(all_data['data']))
else:
	question_order=range(len(all_data['data']))
for outid in range(len(all_data['data'])):
	ii=question_order[outid]
	data=all_data['data'][ii]
# for (ii,data) in enumerate(all_data['data']):
	if outid % 1000==0 and outid!=0:
		print('proc data',outid)
		# break
	passage_text=token_to_text(data['document'])
	if concat_mode=='concat' or (data['question'].count('_')!=1):
		concat_count+=1
		for aid in range(choice_num):

			texts.append(passage_text)
			thishyp=' '.join([token_to_text(data['question']),token_to_text(data['options'][aid])])
			hyps.append(thishyp)
			ids.append(data['id'])
			if aid==data['answer']:
				labels.append(1)
				for rep in range(true_repeat-1):
					texts.append(texts[-1])
					hyps.append(hyps[-1])
					ids.append(ids[-1])
					labels.append(1)
					if verbose:
						print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
						input('check')
			else:
				labels.append(0)
			if verbose:
				print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
				input('check')
	else:
		for aid in range(choice_num):
			texts.append(passage_text)
			unline_pos=data['question'].index('_')

			this_token=data['question'][0:unline_pos]+data['options'][aid]+data['question'][unline_pos+1:]
			thishyp=token_to_text(this_token)
			hyps.append(thishyp)
			ids.append(data['id'])
			if aid==data['answer']:
				labels.append(1)
				for rep in range(true_repeat-1):
					texts.append(texts[-1])
					hyps.append(hyps[-1])
					ids.append(ids[-1])
					labels.append(1)
					if verbose:
						print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
						input('check')
			else:
				labels.append(0)
			if verbose:
				print('question=',data['question'],'choice=',data['options'][aid],)
				print('label=%d, text=%s, hyp=%s' % (labels[-1],texts[-1],hyps[-1]))
				input('check')



if shuffle:
	perm=np.random.permutation(len(hyps))
else:
	perm=range(len(hyps))
tabcount=0
for iidd in range(len(hyps)):
	i=perm[iidd]
	if '\t' in texts[i] or '\t' in hyps[i]:
		tabcount+=1
	# if iidd == 20000:
	# 	break
	print('%d\t%s\t%s\t%s' % (labels[i], texts[i],hyps[i],ids[i]),file=output_file)	
print('tab count=',tabcount)
text_lens=[len(a) for a in texts]
hyp_lens=[len(a) for a in hyps]

n_words_text=[]
n_words_hyp=[]
print('total number of instances:',len(hyps))
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






