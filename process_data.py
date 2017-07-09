import csv
import spacy
import en_core_web_sm as NlpEnglish

parser=NlpEnglish.load(parser=False,matcher=False)
part='test'
n_class=2

input_file=open(r'D:\users\t-yicxu\data\snli_1.0\snli_1.0_'+part+'.txt',encoding='utf-8')
if n_class==3:
	output_file=open(r'D:\users\t-yicxu\data\snli_1.0\\'+part+'.tsv','w',newline='\n',encoding='utf-8')	
else:
	assert n_class==2
	output_file=open(r'D:\users\t-yicxu\data\snli_1.0\\'+part+'_2class.tsv','w',newline='\n',encoding='utf-8')
reader=csv.DictReader(input_file,dialect='excel-tab')
# writer=csv.DictWriter(output_file,dialect='excel-tab')
texts=[]
hyps=[]
labels=[]
ids=[]

counter=0
for line in reader:
	# print(line)
	try:
		if line['gold_label']=='entailment':
			label=1
		elif line['gold_label']=='contradiction':
			if n_class==3:
				label=2
			else:
				label=0
		elif line['gold_label']=='neutral':
			if n_class==3:
				label=3
			else:
				label=0
		else:
			assert line['gold_label']=='-'
			counter+=1
			continue
	except AssertionError:
		print(line['gold_label'])
		input('check')

	# print('%d\t%s\t%s\t%s' % (label, line['sentence1'],line['sentence2'],line['pairID']))
	texts.append(line['sentence1'])
	hyps.append(line['sentence2'])
	labels.append(label)
	ids.append(line['pairID'])
print('%d/%d sentences with no majority.' % (counter,len(labels)))
proc_hyps=parser.pipe(hyps,n_threads=10)
proc_texts=parser.pipe(texts,n_threads=10)
for i in range(len(labels)):
	thishyp=next(proc_hyps)
	thistext=next(proc_texts)
	hypstr=' '.join([d.orth_ for d in thishyp])
	textstr=' '.join([d.orth_ for d in thistext])
	print('%d\t%s\t%s\t%s' % (labels[i], textstr,hypstr,ids[i]),file=output_file)	
