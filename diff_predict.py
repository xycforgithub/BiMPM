import json

dev_data=open(r'D:\users\t-yicxu\data\squad\original\dev-v1.1.json',encoding='utf-8')
pred1_file=open('pred.json',encoding='utf-8')
pred2_file=open('pred_original.json',encoding='utf-8')
out_file=open('../model_data/res.txt','w',encoding='utf-8')

data=json.load(dev_data)
pred1=json.load(pred1_file)
pred2=json.load(pred2_file)
counter=0
longer_counter=0
for item in data['data']:
	for para_data in item['paragraphs']:
		# print(para_data.keys())
		toout={}
		for qa in para_data['qas']:
			ans1=pred1[qa['id']]
			ans2=pred2[qa['id']]
			if ans1!=ans2:
				toout={'question':qa['question'],'gt':qa['answers'],'ans1':ans1,'ans2':ans2,'passage':para_data['context']}
				json.dump(toout,out_file,indent=4)
				print('',file=out_file)
				counter+=1
			if len(ans1)>len(ans2):
				longer_counter+=1
	# print(item['paragraphs'],file=out_file)
print('%d different answers, %d longer' % (counter,longer_counter))



