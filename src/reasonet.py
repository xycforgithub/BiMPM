import tensorflow as tf

import match_utils
from math import log


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def exp_mask(val,mask, name=None):
    return tf.add(val, (1 - mask) * VERY_NEGATIVE_NUMBER, name=name or 'mask')
class ReasoNetModule:
    def __init__(self, num_steps,num_options, state_dim, memory_dim, hidden_dim, lambda_multiplier, memory_max_len, scope=None,
                 terminate_mode='original', logit_combine='sum', keep_first=False):
        # terminate_mode: original or softmax
        # Logit combine: sum or max_pooling
        self.num_steps=num_steps
        self.num_options=num_options
        self.hidden_dim=hidden_dim
        self.state_dim=state_dim
        self.logit_combine=logit_combine
        print('state dim=', state_dim, 'memory dim=',memory_dim)
        self.memory_dim=memory_dim
        self.terminate_mode=terminate_mode
        logit_dim = 2 if terminate_mode=='original' else 1
        with tf.variable_scope(scope or 'reasonet'):
            self.W_state=tf.get_variable('transform_state',[state_dim,hidden_dim],tf.float32)
            self.W_mem=tf.get_variable('transform_memory',[memory_dim,hidden_dim],tf.float32)
            self.W_gate=tf.get_variable('gate_weight',[state_dim,logit_dim], tf.float32)
            self.b_gate=tf.get_variable('gate_bias',[logit_dim], tf.float32)
        self.cell=tf.contrib.rnn.GRUCell(self.state_dim, reuse=tf.get_variable_scope().reuse)
        # print(self.cell.state_size)
        init_state=self.cell.zero_state(50,tf.float32)
        # print(init_state.get_shape())
        # input('check?')
        # self.state_shape=None
        # self.memory_shape=None
        self.memory_max_len=memory_max_len
        self.lambda_multiplier=lambda_multiplier
        self.keep_first=keep_first

        self.total_calculated_steps=self.num_steps+1 if keep_first else self.num_steps
        self.test_vectors=[]
        self.added_1=False
        self.added_2=False

    def multiread_matching(self, matchers, memory):
        tiled_memory_mask=memory.tiled_memory_mask # [batch_size, memory_length]
        all_log_probs=[]
        all_states=[]
        tiled_memory_repre=tf.tile(memory.get_memory_repre(),[self.num_options,1,1]) # [batch_size, memory_length, memory_dim]
        # self.memory_shape=tf.shape(tiled_memory_repre)
        # self.state_shape=tf.shape(matchers[0].aggregation_representation)
        tiled_mapped_memory=match_utils.map_tensor(self.W_mem, tiled_memory_repre, self.memory_dim, [-1, self.memory_max_len, 
                        self.hidden_dim]) # [batch_size, memory_length, hidden_dim]
        # tiled_mapped_memory=tf.tile(mapped_memory,[self.num_options,1,1]) #[batch_size, memory_length, hidden_dim]
        
        cur_states=[]
        num_matcher=len(matchers)
        for matcher in matchers:
            cur_states.append(matcher.aggregation_representation)
        if self.keep_first:
            all_states.extend(cur_states)
        print('num matchers:', len(cur_states))
        for step in range(self.num_steps):
            for mid,state in enumerate(cur_states):
                print('step',step,'matcher',mid)
                input_vector=self.cal_attention_vector(state,tiled_memory_repre, tiled_mapped_memory,tiled_memory_mask) # [batch_size, memory_dim]
                if step>0 or mid>0: 
                    tf.get_variable_scope().reuse_variables()
                # print(input_vector.get_shape())
                # print(state.get_shape())
                _, new_state=self.cell(input_vector, state) # [batch_size, state_dim]
                # print(new_state.get_shape())
                cur_states[mid]=new_state
                all_states.append(new_state)
        print('number of total states:',len(all_states))
        if self.terminate_mode=='original':
            all_terminate_log_probs=[]
            for state in all_states:
                if self.logit_combine=='sum':
                    this_logit=tf.matmul(state, self.W_gate) # [batch_size, 2]
                    if not self.added_2:
                        self.test_vectors.append(this_logit)
                    this_logit=tf.reduce_sum(tf.reshape(this_logit,[self.num_options,-1,2]),axis=0)+self.b_gate # [batch_size/4, 2]
                    if not self.added_2:
                        self.test_vectors.append(this_logit)
                        self.added_2=True
                else:
                    reshaped_state=tf.reshape(state,[self.num_options,-1,self.state_dim]) # [4, batch_size/4, state_dim]
                    max_state=tf.reduce_max(reshaped_state,axis=0)
                    this_logit=tf.matmul(max_state,self.W_gate)+self.b_gate # [batch_size/4, 2]

                all_terminate_log_probs.append(tf.nn.log_softmax(this_logit))
            cur_remaining_log_prob=[]
            for step in range(self.total_calculated_steps):
                for mid in range(num_matcher):
                    stop_prob, go_prob = tf.unstack(all_terminate_log_probs[step * num_matcher + mid], axis=1) # [batch_size/4, 1]
                    self.test_vectors.append(stop_prob)
                    self.test_vectors.append(go_prob)
                    if len(cur_remaining_log_prob)<num_matcher:
                        if self.total_calculated_steps==1:
                            all_log_probs.append(tf.zeros_like(stop_prob))
                        else:
                            all_log_probs.append(stop_prob)
                            cur_remaining_log_prob.append(go_prob)
                    else:
                        if step==self.total_calculated_steps-1:
                            all_log_probs.append(cur_remaining_log_prob[mid])
                        else:
                            all_log_probs.append(cur_remaining_log_prob[mid] + stop_prob)
                            cur_remaining_log_prob[mid]=cur_remaining_log_prob[mid] + go_prob
            all_log_probs=tf.stack(all_log_probs,axis=0) # [num_steps * num_matchers, batch_size/4]
            all_log_probs=tf.reshape(all_log_probs,[self.total_calculated_steps, num_matcher, -1])
            all_states = tf.concat(all_states, axis=0)  # [num_steps * num_matchers * batch_size, state_dim]
        else:
            all_states=tf.concat(all_states,axis=0) # [num_steps * num_matchers * batch_size, state_dim]
            if self.logit_combine=='sum':
                all_logits = tf.matmul(all_states,self.W_gate)
                all_logits = tf.reduce_sum(tf.reshape(all_logits, [self.total_calculated_steps , num_matcher, self.num_options, -1]), axis=2) + self.b_gate  # [num_steps , num_matcher,  batch_size/4]
            else:
                reshaped_state = tf.reshape(all_states, [self.total_calculated_steps , num_matcher, self.num_options, -1, self.state_dim])  # [num_steps , num_matcher, 4, batch_size/4, state_dim]
                max_state = tf.reduce_max(reshaped_state, axis=2) # [num_steps , num_matcher, batch_size/4, state_dim]
                all_logits =tf.matmul(tf.reshape(max_state,[-1,self.state_dim]), self.W_gate)+self.b_gate# [num_steps , num_matcher, batch_size/4]
                all_logits = tf.reshape(all_logits, [self.total_calculated_steps, num_matcher, -1])
            all_log_probs = tf.nn.log_softmax(all_logits, dim=0),[self.total_calculated_steps * num_matcher,-1]  # [num_steps * num_matchers, batch_size/4]


                # all_logits=[]
            # for state in all_states:
            #     all_logits.append(tf.matmul(state,self.W_state)+self.b_gate) # each: [batch_size,1]
            # all_logits=tf.concat(all_logits,axis=1)# [batch_size, num_steps * num_matchers]
            # all_logits=tf.reshape(all_logits,[-1,self.total_calculated_steps,num_matcher])# [batch_size, num_steps, num_matchers]
            # all_log_probs=tf.transpose(all_log_probs)# [num_steps * num_matchers, batch_size]
        # all_states=tf.stack(all_states,axis=0)

        # all_log_probs=tf.reshape(all_log_probs,[num_matcher*self.total_calculated_steps, -1, self.num_options])
        # all_log_probs=tf.reduce_logsumexp(all_log_probs, axis=2)-log(self.num_options)
        print('finished reasonet')
        return all_log_probs,all_states
    # def cal_multiread_result(self, all_states,w_0,b_0,w_1,b_1,is_training,use_options=True, num_options=4, layout='choice_first')):







    def cal_attention_vector(self,cur_state, memory_repre, mapped_memory, memory_mask):
        # mapped_memory=match_utils.map_tensor(self.W_mem, memory_repre, self.memory_dim, [-1, self.memory_max_len, self.hidden_dim]) # [batch_size, memory_length, hidden_dim]
        # mapped_state=self.map_tensor(self.W_state, cur_state, self.state_dim, self.state_shape)
        mapped_state=tf.matmul(cur_state,self.W_state) # [batch_size, hidden_dim]
        expanded_state=tf.expand_dims(mapped_state,axis=1) # [batch_size, 1, hidden_dim]
        relevancy_mat=tf.squeeze(match_utils.cal_relevancy_matrix(expanded_state,mapped_memory), axis=2)*self.lambda_multiplier # [batch_size, memory_length]
        # print('relevancy_mat',relevancy_mat.get_shape())
        relevancy_mat=exp_mask(relevancy_mat,memory_mask) # [batch_size, memory_length]
        softmax_sim=tf.nn.softmax(relevancy_mat, name='reasonet_attention_softmax')# [batch_size, memory_length]
        # print(softmax_sim.name)
        # print('softmax_sim',softmax_sim.get_shape())
        res = tf.multiply(memory_repre,tf.expand_dims(softmax_sim,axis=2))# [batch_size, memory_length, memory_dim]
        res=tf.reduce_sum(res,axis=1)
        if not self.added_1:
            self.test_vectors.extend([mapped_memory,mapped_state,relevancy_mat,softmax_sim,res])
            self.added_1=True
        
        return res # [batch_size, memory_dim]




