import numpy as np

pred_file=open(r'd:\users\t-yicxu\model_data\BiMPM\SentenceMatch.race_replace_middle.probs')
n_choice=4
label_list=[]
prob_list=[]
pbcount=0
correctcount=0
for entail_line in pred_file:
	try:
		choice,entail_probs=entail_line.split('\t')
	except ValueError:
		print(entail_line)
		input('check')
	# print(entail_line)
	# entail_probs=entail_line
	find=False
	for each_prob in entail_probs.split(' '):
		label,prob = each_prob.split(':')
		if label=='1':
			this_entail_prob=float(prob)
			find=True
	assert find
	label_list.append(int(choice))
	prob_list.append(this_entail_prob)
	if len(label_list)==n_choice:
		pred_ans=np.argmax(prob_list)
		gt_ans=np.argmax(label_list)
		assert(int(label_list[gt_ans])==1)
		if pred_ans==gt_ans:
			correctcount+=1
		pbcount+=1
		# print(pred_ans,gt_ans)
		# input('check')
		label_list=[]
		prob_list=[]
print('accuracy:',correctcount/pbcount)
