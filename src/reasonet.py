import tensorflow as tf

import match_utils


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def exp_mask(val,mask, name=None):
    return tf.add(val, (1 - mask) * VERY_NEGATIVE_NUMBER, name=name or 'mask')
class ReasoNetModule:
    def __init__(self, num_steps,num_options, state_dim, memory_dim, hidden_dim, lambda_multiplier, scope=None,
                 terminate_mode='original', keep_first=False):
        # terminate_mode: original or softmax
        self.num_steps=num_steps
        self.num_options=num_options
        self.hidden_dim=hidden_dim
        self.state_dim=state_dim
        self.memory_dim=memory_dim
        self.terminate_mode=terminate_mode
        logit_dim = 2 if terminate_mode=='original' else 1
        with tf.variable_scope(scope or 'reasonet'):
            self.W_state=tf.get_variable('transform_state',[state_dim,hidden_dim],tf.float32)
            self.W_mem=tf.get_variable('transform_memory',[memory_dim,hidden_dim],tf.float32)
            self.W_gate=tf.get_variable('gate_weight',[state_dim,logit_dim], tf.float32)
            self.b_gate=tf.get_variable('gate_bias',[logit_dim], tf.float32)
        self.cell=tf.contrib.rnn.GRUCell(self.state_dim)
        self.state_shape=None
        self.memory_shape=None
        self.lambda_multiplier=lambda_multiplier
        self.keep_first=keep_first

        self.total_calculated_steps=self.num_steps+1 if keep_first else self.num_steps

    def multiread_matching(self, matchers, memory):
        tiled_memory_mask=memory.tiled_memory_mask # [batch_size, memory_length]
        tiled_memory_repre=tf.tile(memory.get_memory_repre(),[self.num_options,1,1]) # [batch_size, memory_length, memory_dim]
        self.memory_shape=tf.shape(tiled_memory_repre)
        self.state_shape=tf.shape(matchers[0].aggregation_representation)
        all_log_probs=[]
        all_states=[]

        cur_states=[]
        num_matcher=len(matchers)
        for matcher in matchers:
            cur_states.append(matcher.aggregation_representation)
        if self.keep_first:
            all_states=cur_states

        for step in range(self.num_steps):
            for mid,state in enumerate(cur_states):
                input_vector=self.cal_attention_vector(state,tiled_memory_repre,tiled_memory_mask) # [batch_size, memory_dim]
                _, new_state=self.cell(input_vector, state) # [batch_size, state_dim]
                cur_states[mid]=new_state
                all_states.append(new_state)
        if self.terminate_mode=='original':
            all_terminate_log_probs=[]
            for state in all_states:
                this_logit=tf.matmul(state, self.W_gate)+self.b_gate # [batch_size, 2]
                all_terminate_log_probs.append(tf.nn.log_softmax(this_logit))
            cur_remaining_log_prob=[]
            for step in range(self.total_calculated_steps):
                for mid in range(num_matcher):
                    stop_prob, go_prob = tf.unstack(all_terminate_log_probs[step * num_matcher + mid], axis=1)
                    if len(cur_remaining_log_prob)<num_matcher:
                        all_log_probs.append(stop_prob)
                        cur_remaining_log_prob.append(go_prob)
                    else:
                        all_log_probs.append(cur_remaining_log_prob[mid] + stop_prob)
                        cur_remaining_log_prob[mid]=cur_remaining_log_prob[mid] * go_prob
            all_log_probs=tf.stack(all_log_probs,axis=0) # [num_steps * num_matchers, batch_size]
        else:
            all_logits=[]
            for state in all_states:
                all_logits.append(tf.matmul(state,self.W_state)+self.b_gate) # each: [batch_size,1]
            all_logits=tf.concat(all_logits,axis=1)# [batch_size, num_steps * num_matchers]
            all_logits=tf.reshape(all_logits,[-1,self.total_calculated_steps,num_matcher])# [batch_size, num_steps, num_matchers]
            all_log_probs=tf.reshape(tf.nn.log_softmax(all_logits,dim=1),[-1,num_matcher*self.num_steps])# [batch_size, num_steps * num_matchers]
            all_log_probs=tf.transpose(all_log_probs)# [num_steps * num_matchers, batch_size]
        # all_states=tf.stack(all_states,axis=0)
        all_states=tf.concat(all_states, axis=0)# [num_steps * num_matchers * batch_size, state_dim]
        return all_log_probs,all_states
    # def cal_multiread_result(self, all_states,w_0,b_0,w_1,b_1,is_training,use_options=True, num_options=4, layout='choice_first')):






    def map_tensor(self, weight, tensor, vector_dim, target_shape):
        reshaped_tensor=tf.reshape(tensor, [-1,vector_dim])
        mapped_reshaped=tf.matmul(reshaped_tensor,weight)
        return tf.reshape(mapped_reshaped, target_shape)
    def cal_attention_vector(self,cur_state, memory_repre, memory_mask):
        mapped_memory=self.map_tensor(self.W_mem, memory_repre, self.memory_dim, self.memory_shape) # [batch_size, memory_length, hidden_dim]
        # mapped_state=self.map_tensor(self.W_state, cur_state, self.state_dim, self.state_shape)
        mapped_state=tf.matmul(cur_state,self.W_state) # [batch_size, hidden_dim]
        expanded_state=tf.expand_dims(mapped_state,axis=1) # [batch_size, 1, hidden_dim]
        relevancy_mat=tf.squeeze(match_utils.cal_relevancy_matrix(expanded_state,mapped_memory))*self.lambda_multiplier # [batch_size, memory_length]
        relevancy_mat=exp_mask(relevancy_mat,memory_mask) # [batch_size, memory_length]
        softmax_sim=tf.nn.softmax(relevancy_mat)# [batch_size, memory_length]
        res = tf.multiply(memory_repre,tf.expand_dims(relevancy_mat,axis=2))# [batch_size, memory_length, memory_dim]
        return tf.reduce_sum(res,axis=1) # [batch_size, memory_dim]




