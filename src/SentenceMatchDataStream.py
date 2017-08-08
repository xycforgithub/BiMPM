import numpy as np
import re
num_options=4
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_matrix(in_val, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in range(batch_size):
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val

def pad_3d_tensor(in_val, max_length1=None, max_length2=None, dtype=np.int32):
    if max_length1 is None: max_length1 = np.max([len(cur_in_val) for cur_in_val in in_val])
    if max_length2 is None: max_length2 = np.max([np.max([len(val) for val in cur_in_val]) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length1, max_length2), dtype=dtype)
    for i in range(batch_size):
        cur_length1 = max_length1
        if len(in_val[i])<max_length1: cur_length1 = len(in_val[i])
        for j in range(cur_length1):
            cur_in_val = in_val[i][j]
            kept_length = len(cur_in_val)
            if kept_length>max_length2: kept_length = max_length2
            out_val[i, j, :kept_length] = cur_in_val[:kept_length]
    return out_val




class SentenceMatchDataStream(object):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, label_vocab=None, batch_size=60, 
                 isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, max_sent_length=200,max_hyp_length=100):
        instances = []
        infile = open(inpath, 'rt',encoding='utf-8')
        for line in infile:
            line = line.strip()
            if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            sentence1 = items[1].lower()
            sentence2 = items[2].lower()
            if label_vocab is not None: 
                label_id = label_vocab.getIndex(label)
                if label_id >= label_vocab.vocab_size: label_id = 0
            else: 
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)
            word_idx_2 = word_vocab.to_index_sequence(sentence2)
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)
            if len(word_idx_1)>max_sent_length: 
                word_idx_1 = word_idx_1[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
            if len(word_idx_2)>max_hyp_length:
                word_idx_2 = word_idx_2[:max_hyp_length]
                char_matrix_idx_2 = char_matrix_idx_2[:max_hyp_length]

            POS_idx_1 = None
            POS_idx_2 = None
            if POS_vocab is not None:
                POS_idx_1 = POS_vocab.to_index_sequence(items[3])
                if len(POS_idx_1)>max_sent_length: POS_idx_1 = POS_idx_1[:max_sent_length]
                POS_idx_2 = POS_vocab.to_index_sequence(items[4])
                if len(POS_idx_2)>max_sent_length: POS_idx_2 = POS_idx_2[:max_sent_length]

            NER_idx_1 = None
            NER_idx_2 = None
            if NER_vocab is not None:
                NER_idx_1 = NER_vocab.to_index_sequence(items[5])
                if len(NER_idx_1)>max_sent_length: NER_idx_1 = NER_idx_1[:max_sent_length]
                NER_idx_2 = NER_vocab.to_index_sequence(items[6])
                if len(NER_idx_2)>max_sent_length: NER_idx_2 = NER_idx_2[:max_sent_length]
            

            instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                              POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2))
        infile.close()

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[4]), len(instance[5]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, batch_size) 
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            char_matrix_idx_1_batch = []
            char_matrix_idx_2_batch = []
            sent1_length_batch = []
            sent2_length_batch = []
            sent1_char_length_batch = []
            sent2_char_length_batch = []

            POS_idx_1_batch = None
            if POS_vocab is not None: POS_idx_1_batch = []
            POS_idx_2_batch = None
            if POS_vocab is not None: POS_idx_2_batch = []

            NER_idx_1_batch = None
            if NER_vocab is not None: NER_idx_1_batch = []
            NER_idx_2_batch = None
            if NER_vocab is not None: NER_idx_2_batch = []

            for i in range(batch_start, batch_end):
                (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2,
                 POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2) = instances[i]
                label_batch.append(label)
                sent1_batch.append(sentence1)
                sent2_batch.append(sentence2)
                label_id_batch.append(label_id)
                word_idx_1_batch.append(word_idx_1)
                word_idx_2_batch.append(word_idx_2)
                char_matrix_idx_1_batch.append(char_matrix_idx_1)
                char_matrix_idx_2_batch.append(char_matrix_idx_2)
                sent1_length_batch.append(len(word_idx_1))
                sent2_length_batch.append(len(word_idx_2))
                sent1_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                sent2_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])

                if POS_vocab is not None: 
                    POS_idx_1_batch.append(POS_idx_1)
                    POS_idx_2_batch.append(POS_idx_2)

                if NER_vocab is not None: 
                    NER_idx_1_batch.append(NER_idx_1)
                    NER_idx_2_batch.append(NER_idx_2)
                
                
            cur_batch_size = len(label_batch)
            if cur_batch_size ==0: continue

            # padding
            max_sent1_length = np.max(sent1_length_batch)
            max_sent2_length = np.max(sent2_length_batch)

            max_char_length1 = np.max([np.max(aa) for aa in sent1_char_length_batch])
            if max_char_length1>max_char_per_word: max_char_length1=max_char_per_word

            max_char_length2 = np.max([np.max(aa) for aa in sent2_char_length_batch])
            if max_char_length2>max_char_per_word: max_char_length2=max_char_per_word
            
            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)

            char_matrix_idx_1_batch = pad_3d_tensor(char_matrix_idx_1_batch, max_length1=max_sent1_length, max_length2=max_char_length1)
            char_matrix_idx_2_batch = pad_3d_tensor(char_matrix_idx_2_batch, max_length1=max_sent2_length, max_length2=max_char_length2)

            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)

            sent1_char_length_batch = pad_2d_matrix(sent1_char_length_batch, max_length=max_sent1_length)
            sent2_char_length_batch = pad_2d_matrix(sent2_char_length_batch, max_length=max_sent2_length)
            
            if POS_vocab is not None:
                POS_idx_1_batch = pad_2d_matrix(POS_idx_1_batch, max_length=max_sent1_length)
                POS_idx_2_batch = pad_2d_matrix(POS_idx_2_batch, max_length=max_sent2_length)
            if NER_vocab is not None:
                NER_idx_1_batch = pad_2d_matrix(NER_idx_1_batch, max_length=max_sent1_length)
                NER_idx_2_batch = pad_2d_matrix(NER_idx_2_batch, max_length=max_sent2_length)
                

            self.batches.append((label_batch, sent1_batch, sent2_batch, label_id_batch, word_idx_1_batch, word_idx_2_batch, 
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, sent1_length_batch, sent2_length_batch, 
                                 sent1_char_length_batch, sent2_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch))
        
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

def gen_concat_indx_mat(batch1_lengths,batch2_lengths):
    # sum_length=np.array(batch1_lengths)+np.array(batch2_lengths)
    sum_length=batch1_lengths+batch2_lengths
    max_length=np.max(sum_length)
    max_length_1=np.max(batch1_lengths)
    max_length_2=np.max(batch2_lengths)
    batch_size=len(batch1_lengths)
    idx=np.zeros((batch_size, max_length,2),dtype=np.int32)
    for i in range(batch_size):
        counter=0
        for j in range(batch1_lengths[i]):
            idx[i,counter,0]=i
            idx[i,counter,1]=j
            counter+=1
        for j in range(batch2_lengths[i]):
            idx[i,counter,0]=i
            idx[i,counter,1]=j+max_length_1
            counter+=1
        for k in range(counter,max_length):
            idx[i,k,0]=i
            idx[i,k,1]=batch2_lengths[i]+max_length_1-1

    return idx,sum_length.tolist()
def gen_split_indx_mat(batch1_lengths,batch2_lengths):
    # sum_length=np.array(batch1_lengths)+np.array(batch2_lengths)
    sum_length=batch1_lengths+batch2_lengths
    max_length=np.max(sum_length)
    max_length_1=np.max(batch1_lengths)
    max_length_2=np.max(batch2_lengths)
    batch_size=len(batch1_lengths)
    idx1=np.zeros((batch_size,max_length_1,2),dtype=np.int32)
    idx2=np.zeros((batch_size,max_length_2,2),dtype=np.int32)
    for i in range(batch_size):
        for j in range(batch1_lengths[i]):
            idx1[i,j,0]=i
            idx1[i,j,1]=j
        for j in range(batch1_lengths[i],max_length_1):
            idx1[i,j,0]=i
            idx1[i,j,1]=batch1_lengths[i]-1
        for j in range(batch2_lengths[i]):
            idx2[i,j,0]=i
            idx2[i,j,1]=batch1_lengths[i]+j
        for j in range(batch2_lengths[i],max_length_2):
            idx2[i,j,0]=i
            idx2[i,j,1]=batch1_lengths[i]+batch2_lengths[i]-1
    return idx1,idx2            

        
class TriMatchDataStream(SentenceMatchDataStream):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None, label_vocab=None, batch_size=60, 
                 isShuffle=False, isLoop=False, isSort=True, max_char_per_word=10, max_sent_length=200,max_hyp_length=100, max_choice_length=None,
                 tolower=False, gen_concat_mat=False, gen_split_mat=False, efficient=False, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        if max_choice_length is None:
            max_choice_length=max_hyp_length
        instances = []
        infile = open(inpath, 'rt',encoding='utf-8')
        for line in infile:
            line = line.strip()
            if line.startswith('-'): continue
            items = re.split("\t", line)
            label = items[0]
            if tolower:
                sentence1 = items[1].lower()
                sentence2 = items[2].lower()
                sentence3 = items[3].lower()
            else:                
                sentence1 = items[1]
                sentence2 = items[2]
                sentence3 = items[3]
            if label_vocab is not None: 
                label_id = label_vocab.getIndex(label)
                if label_id >= label_vocab.vocab_size: label_id = 0
            else: 
                label_id = int(label)
            word_idx_1 = word_vocab.to_index_sequence(sentence1)
            word_idx_2 = word_vocab.to_index_sequence(sentence2)
            word_idx_3 = word_vocab.to_index_sequence(sentence3)
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2)
            char_matrix_idx_3 = char_vocab.to_character_matrix(sentence3)
            if len(word_idx_1)>max_sent_length: 
                word_idx_1 = word_idx_1[:max_sent_length]
                char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
            if len(word_idx_2)>max_hyp_length:
                word_idx_2 = word_idx_2[:max_hyp_length]
                char_matrix_idx_2 = char_matrix_idx_2[:max_hyp_length]
            if len(word_idx_3)>max_choice_length:
                word_idx_3 = word_idx_3[:max_choice_length]
                char_matrix_idx_3 = char_matrix_idx_3[:max_choice_length]
            POS_idx_1 = None
            POS_idx_2 = None
            if POS_vocab is not None:
                POS_idx_1 = POS_vocab.to_index_sequence(items[4])
                if len(POS_idx_1)>max_sent_length: POS_idx_1 = POS_idx_1[:max_sent_length]
                POS_idx_2 = POS_vocab.to_index_sequence(items[5])
                if len(POS_idx_2)>max_sent_length: POS_idx_2 = POS_idx_2[:max_sent_length]

            NER_idx_1 = None
            NER_idx_2 = None
            if NER_vocab is not None:
                NER_idx_1 = NER_vocab.to_index_sequence(items[6])
                if len(NER_idx_1)>max_sent_length: NER_idx_1 = NER_idx_1[:max_sent_length]
                NER_idx_2 = NER_vocab.to_index_sequence(items[7])
                if len(NER_idx_2)>max_sent_length: NER_idx_2 = NER_idx_2[:max_sent_length]
            

            instances.append((label, sentence1, sentence2,sentence3, label_id, word_idx_1, word_idx_2,word_idx_3, 
                        char_matrix_idx_1, char_matrix_idx_2, char_matrix_idx_3, POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2))
        infile.close()

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[5]), len(instance[6]),len(instance[7]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, batch_size) 
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_batch = []
            sent1_batch = []
            sent2_batch = []
            sent3_batch = []
            label_id_batch = []
            word_idx_1_batch = []
            word_idx_2_batch = []
            word_idx_3_batch = []
            char_matrix_idx_1_batch = []
            char_matrix_idx_2_batch = []
            char_matrix_idx_3_batch = []
            sent1_length_batch = []
            sent2_length_batch = []
            sent3_length_batch = []
            sent1_char_length_batch = []
            sent2_char_length_batch = []
            sent3_char_length_batch = []
            split_mat_batch_q=None
            split_mat_batch_c=None
            concat_mat_batch=None

            POS_idx_1_batch = None
            if POS_vocab is not None: POS_idx_1_batch = []
            POS_idx_2_batch = None
            if POS_vocab is not None: POS_idx_2_batch = []

            NER_idx_1_batch = None
            if NER_vocab is not None: NER_idx_1_batch = []
            NER_idx_2_batch = None
            if NER_vocab is not None: NER_idx_2_batch = []

            for i in range(batch_start, batch_end):
                (label, sentence1, sentence2,sentence3, label_id, word_idx_1, word_idx_2,word_idx_3, char_matrix_idx_1, char_matrix_idx_2,
                        char_matrix_idx_3, POS_idx_1, POS_idx_2, NER_idx_1, NER_idx_2) = instances[i]
                if (not efficient) or i % 4==0:
                    sent1_batch.append(sentence1)
                    sent2_batch.append(sentence2)
                    word_idx_1_batch.append(word_idx_1)
                    word_idx_2_batch.append(word_idx_2)
                    char_matrix_idx_1_batch.append(char_matrix_idx_1)
                    char_matrix_idx_2_batch.append(char_matrix_idx_2)
                    sent1_length_batch.append(len(word_idx_1))
                    sent2_length_batch.append(len(word_idx_2))
                    sent1_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                    sent2_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])
                
                sent3_batch.append(sentence3)
                label_batch.append(label)
                label_id_batch.append(label_id)
                word_idx_3_batch.append(word_idx_3)
                char_matrix_idx_3_batch.append(char_matrix_idx_3)
                sent3_length_batch.append(len(word_idx_3))
                sent3_char_length_batch.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_3])


                # if POS_vocab is not None: 
                #     POS_idx_1_batch.append(POS_idx_1)
                #     POS_idx_2_batch.append(POS_idx_2)

                # if NER_vocab is not None: 
                #     NER_idx_1_batch.append(NER_idx_1)
                #     NER_idx_2_batch.append(NER_idx_2)
            cur_batch_size = len(label_batch)
            if cur_batch_size ==0: continue
    
            if efficient:
                num_questions=cur_batch_size//num_options
                idx_list=[]
                for optid in range(num_options):
                    for qid in range(num_questions):
                        idx_list.append(qid*num_options+optid)
                sent3_batch=[sent3_batch[i] for i in idx_list]
                label_batch=[label_batch[i] for i in idx_list]
                label_id_batch=[label_id_batch[i] for i in idx_list]
                word_idx_3_batch=[word_idx_3_batch[i] for i in idx_list]
                char_matrix_idx_3_batch=[char_matrix_idx_3_batch[i] for i in idx_list]
                sent3_length_batch=[sent3_length_batch[i] for i in idx_list]
                sent3_char_length_batch=[sent3_char_length_batch[i] for i in idx_list]
            
            # padding
            max_sent1_length = np.max(sent1_length_batch)
            max_sent2_length = np.max(sent2_length_batch)
            max_sent3_length = np.max(sent3_length_batch)


            max_char_length1 = np.max([np.max(aa) for aa in sent1_char_length_batch])
            if max_char_length1>max_char_per_word: max_char_length1=max_char_per_word

            max_char_length2 = np.max([np.max(aa) for aa in sent2_char_length_batch])
            if max_char_length2>max_char_per_word: max_char_length2=max_char_per_word
            
            max_char_length3 = np.max([np.max(aa) for aa in sent3_char_length_batch])
            if max_char_length3>max_char_per_word: max_char_length3=max_char_per_word

            label_id_batch = np.array(label_id_batch)
            word_idx_1_batch = pad_2d_matrix(word_idx_1_batch, max_length=max_sent1_length)
            word_idx_2_batch = pad_2d_matrix(word_idx_2_batch, max_length=max_sent2_length)
            word_idx_3_batch = pad_2d_matrix(word_idx_3_batch, max_length=max_sent3_length)

            char_matrix_idx_1_batch = pad_3d_tensor(char_matrix_idx_1_batch, max_length1=max_sent1_length, max_length2=max_char_length1)
            char_matrix_idx_2_batch = pad_3d_tensor(char_matrix_idx_2_batch, max_length1=max_sent2_length, max_length2=max_char_length2)
            char_matrix_idx_3_batch = pad_3d_tensor(char_matrix_idx_3_batch, max_length1=max_sent3_length, max_length2=max_char_length3)


            sent1_length_batch = np.array(sent1_length_batch)
            sent2_length_batch = np.array(sent2_length_batch)
            sent3_length_batch = np.array(sent3_length_batch)

            if gen_concat_mat:
                if efficient:
                    tiled_sent2_length_batch=np.tile(sent2_length_batch,num_options)
                else:
                    tiled_sent2_length_batch=sent2_length_batch
                concat_mat_batch, _=gen_concat_indx_mat(tiled_sent2_length_batch,sent3_length_batch)
                if gen_split_mat:
                    split_mat_batch_q, split_mat_batch_c=gen_split_indx_mat(tiled_sent2_length_batch,sent3_length_batch)

            sent1_char_length_batch = pad_2d_matrix(sent1_char_length_batch, max_length=max_sent1_length)
            sent2_char_length_batch = pad_2d_matrix(sent2_char_length_batch, max_length=max_sent2_length)
            sent3_char_length_batch = pad_2d_matrix(sent3_char_length_batch, max_length=max_sent3_length)
            
            if POS_vocab is not None:
                POS_idx_1_batch = pad_2d_matrix(POS_idx_1_batch, max_length=max_sent1_length)
                POS_idx_2_batch = pad_2d_matrix(POS_idx_2_batch, max_length=max_sent2_length)
            if NER_vocab is not None:
                NER_idx_1_batch = pad_2d_matrix(NER_idx_1_batch, max_length=max_sent1_length)
                NER_idx_2_batch = pad_2d_matrix(NER_idx_2_batch, max_length=max_sent2_length)
                

            self.batches.append((label_batch, sent1_batch, sent2_batch, sent3_batch, label_id_batch,
                                 word_idx_1_batch, word_idx_2_batch, word_idx_3_batch,
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, char_matrix_idx_3_batch, 
                                 sent1_length_batch, sent2_length_batch, sent3_length_batch,
                                 sent1_char_length_batch, sent2_char_length_batch, sent3_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch,
                                 concat_mat_batch,split_mat_batch_q, split_mat_batch_c))
        
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
