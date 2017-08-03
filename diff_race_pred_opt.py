import numpy as np
import random
import json
import scipy.stats
import csv
from collections import defaultdict



test_mode='test'
subset='all'
pred_file1=open(r'D:\users\t-yicxu\model_data\BiMPM\race\TriMatch.race_tri_rl_contrastive_imp_concat.iter34957.probs')
pred_file2=open(r'd:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_concat_opt_middle_noright2_test.probs')
out_file=open('../model_data/res_diff_race3.txt','w',encoding='utf-8')
match_ids=True
# input_data2=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_high.json',encoding='utf-8')
# input_data=open(r'D:\users\t-yicxu\data\race\processed\\'+test_mode+'_middle.json',encoding='utf-8')
# if subset=='high':
# 	all_data=json.load(input_data2)
# elif subset=='middle':
# 	all_data=json.load(input_data)
# else:
# 	all_data=json.load(input_data)
# 	data2=json.load(input_data2)
# 	all_data['data']+=data2['data']



output_count=0
# out_file=open(r'../model_data/result.txt','w',encoding='utf-8')
data=json.load(open('res.txt'))
corrects=data['correct']
totals=data['total']

n_choice=4
label_list=[]
prob_list=[]
pbcount=0
correctcount1=0
correctcount2=0
either_correct_count=0
diffcount=0
diffcount_raw=0
vote_correct_count=0
buffers=[]
subcount=0
subtotcount=0
pointer=0
sum_kl=0.0
def process_entail_line(entail_line):
	choice,entail_probs=entail_line.split('\t')
	label_pbs=[]
	for each_prob in entail_probs.split(' '):
		label,prob = each_prob.split(':')
		label_pbs.append(float(prob))
	prediction=np.argmax(label_pbs)
	return int(choice),label_pbs,prediction
all_lines_file2=[]
for i,line in enumerate(pred_file2):
	all_lines_file2.append(line)
	choice2,label_pbs2,prediction2=process_entail_line(line)
	# if i<=20:
	# 	print(choice2,end=' ')
# print('')
if match_ids:
	data_file1=open(r'D:\users\t-yicxu\data\race_clean\entail_test_concat_options_tri_middle_sorted.tsv', encoding='utf-8')
	data_file2=open(r'D:\users\t-yicxu\data\race\entail_test_concat_middle_sorted.tsv', encoding='utf-8')

	fieldnames=['label','passage','question','choice','id']
	fieldnames2=['label','passage','question','id']
	reader1=csv.DictReader(data_file1,fieldnames,dialect='excel-tab')
	reader2=csv.DictReader(data_file2,fieldnames2,dialect='excel-tab')
	mappingdict={}
	mappinglabellist=defaultdict(list)
	mappingdatalist=defaultdict(list)
	all_data_list1=[]
	counter=0
	for i,row in enumerate(reader1):
		# print(row)
		mappinglabellist[row['id']].append(int(row['label']))
		mappingdatalist[row['id']].append(row)
		if len(mappingdatalist[row['id']])==4:
			all_data_list1.append(mappingdatalist[row['id']])
		if i % 4 ==0:
			assert row['id'] not in mappingdict
			mappingdict[row['id']]=counter
			# print(row['id'])
			counter+=1
			# input('check')
	mappinglist=[0 for i in range(len(all_lines_file2))]
	cur_list=[]
	all_data_list2=[]
	cur_data_list=[]
	for i,row in enumerate(reader2):
		cur_list.append(int(row['label']))
		cur_data_list.append(row)
		# print(row)
		if i % 4 ==0:
			mappinglist[mappingdict[row['id']]]=i//4
			# mappinglist.append(mappingdict[row['id']])
		if len(cur_list)==4:
			assert np.sum(cur_list)==1
			all_data_list2.append(cur_data_list)
			cur_data_list=[]
			# print(np.argmax(cur_list),end=' ')
			assert np.argmax(cur_list)==np.argmax(mappinglabellist[row['id']])
			cur_list=[]

else:
	mappinglist=range(len(all_lines_file2))


for lid,entail_line1 in enumerate(pred_file1):


	choice1,label_pbs1,prediction1=process_entail_line(entail_line1)
	entail_line2=all_lines_file2[mappinglist[lid]]
	choice2,label_pbs2,prediction2=process_entail_line(entail_line2)
	if choice1!=choice2:
		print(choice1,choice2)
		print(all_data_list1[lid])
		print(all_data_list2[mappinglist[lid]])
	assert choice1==choice2
	# print('correct')
	pbcount+=1
	if prediction1==choice1:
		correctcount1+=1
	if prediction2==choice1:
		correctcount2+=1
	if prediction2==choice1 and prediction1==choice1:
		either_correct_count+=1
	if prediction1!=prediction2:
		diffcount_raw+=1
	if prediction1!=prediction2 and (prediction1==choice1 or prediction2==choice2):
		diffcount+=1
	if prediction2==choice1 and prediction1!=choice1:
	# if choice1==choice2:
		this_data=all_data_list1[lid]
		choices=[this_data[tmpid]['choice'] for tmpid in range(4)]
		print('passage:',all_data_list1[lid][0]['passage'],'question:',all_data_list1[lid][0]['question'],
			'choice:',choices, 'gt and pred1=',choice1, 'pred2=',prediction2, file=out_file)
		output_count+=1

	vote_pbs=[label_pbs1[i]+label_pbs2[i] for i in range(len(label_pbs1))]
	vote_pred=np.argmax(vote_pbs)
	if vote_pred==choice1:
		vote_correct_count+=1
	sum_kl+=scipy.stats.entropy(label_pbs1,label_pbs2)





print('accuracy1: ',correctcount1/pbcount*100)
print('accuracy2: ',correctcount2/pbcount*100)
print('Either correct accuracy: ',either_correct_count/pbcount*100)
print('raw diff ratio: ',diffcount_raw/pbcount*100)
print('meaningful diff ratio: ',diffcount/pbcount*100)
print('vote accuracy: ',vote_correct_count/pbcount*100)
print('average KL divergence:', sum_kl/pbcount)
print('output_count:',output_count,'/',pbcount)