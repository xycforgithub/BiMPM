import numpy as np
import re
def collect_vocabs(train_path):
    all_words = set()

    infile = open(train_path, 'rt',encoding='utf-8')
    for (i,line) in enumerate(infile):
        if i % 10000==0:
            print('input line',i)
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        sentence1 = re.split("\\s+",items[1].strip().lower())
        sentence2 = re.split("\\s+",items[2].strip().lower())
        all_words.update(sentence1)
        all_words.update(sentence2)
        if '' in sentence1:
            print(items[1].lower())
            print(sentence1)
            input('check')
        if '' in sentence2:
            print(items[2].lower())
            print(sentence2)
            input('check')
        

    infile.close()

    return (all_words)
def fromText_format3(vec_path,voc=None):
    # load freq table and build index for each word
    word2id = {}
    id2word = {}

    vec_file = open(vec_path, 'rt',encoding='utf-8')
    
    word_dim=0
    word_vecs = {}
    for (i,line) in enumerate(vec_file):
        if i % 100000==0:
            print('vec line',i)
        line = line.strip()
        parts = line.split(' ')
        word = parts[0]
        if len(parts[1:])!=word_dim and word_dim!=0:
            continue
        else:
            word_dim = len(parts[1:])
            if (voc is not None) and (word not in voc): continue
            vector = np.array(parts[1:], dtype='float32')
        cur_index = len(word2id)
        word2id[word] = cur_index 
        id2word[cur_index] = word
        word_vecs[cur_index] = vector
    vec_file.close()

#     vocab_size = len(word2id)
#     final_word_vecs = np.zeros((vocab_size, word_dim), dtype=np.float32) # the last dimension is all zero
#     try:
#         for cur_index in range(vocab_size):
#             final_word_vecs[cur_index] = word_vecs[cur_index]
#     except ValueError:
#         print(line)
#         print('word=',id2word[cur_index])
#         print('vector=',word_vecs[cur_index])
#         print('shape=',word_vecs[cur_index].shape)
#         input('check')
    return word_vecs,word2id,id2word
# train_path=r'D:\users\t-yicxu\data\snli_1.0\train.tsv'
# dev_path=r'D:\users\t-yicxu\data\snli_1.0\dev.tsv'
# test_path=r'D:\users\t-yicxu\data\snli_1.0\test.tsv'
vec_path=r'D:/users/t-yicxu/biglearn/glove.840B.300d.txt'

# train_path=r'D:\users\t-yicxu\data\race\entail_train_concat.tsv'
# dev_path=r'D:\users\t-yicxu\data\race\entail_dev_concat.tsv'


# train_path=r'D:\users\t-yicxu\data\squad\entail_train_4_0_f1_2class_re3.tsv'
# dev_path=r'D:\users\t-yicxu\data\squad\entail_dev_1_0_f1_2class_re1.tsv'
# test_path=r'D:\users\t-yicxu\data\squad\entail_dev_same_1_overlap_2class.tsv'

# train_path_2=r'D:\users\t-yicxu\data\snli_1.0\train.tsv'

train_path=r'D:\users\t-yicxu\data\squad\entail_test_10choice.tsv'

print('start')
all_words=collect_vocabs(train_path)
# Uncomment this to add more words in dev and test
# all_words|= collect_vocabs(train_path_2)
# all_words|= collect_vocabs(dev_path)
# all_words|=collect_vocabs(test_path)

print('%d words in total.'% len(all_words))

word_vecs,word2id,id2word=fromText_format3(vec_path)
# print('finished loading')
# vec_out=open(r'D:\users\t-yicxu\data\snli_1.0\word2vec_test.txt','w',encoding='utf-8',newline='\n')
# vec_out=open(r'D:\users\t-yicxu\data\race\word2vec_temp.txt','w',encoding='utf-8',newline='\n')
vec_out=open(r'D:\users\t-yicxu\data\squad\word2vec_entail_test_10choice.txt','w',encoding='utf-8',newline='\n')
vec_dim=word_vecs[0].shape[0]

n=len(word2id)
counter=0
for (i,word) in enumerate(all_words):
    if i % 1000==0:
        print('word', i)
    if word in word2id:
        vec=word_vecs[word2id[word]]
        counter+=1
    else:
        continue
        vec=np.random.normal(scale=0.3,size=(vec_dim))    
    vec_str=' '.join([t for t in vec.astype(str)])
    print('%s %s' % (word,vec_str), file=vec_out)
vec_out.close()
print('finished',counter,'words')