import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn
from match_utils import multi_highway_layer
import match_utils
from matcher import Matcher
from memory import Memory
from tensorflow.python.ops.rnn_cell_impl import _linear

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

    def multiread_matching(self, matchers, memory, memory_mask):
        num_matcher=len(matchers)
        tiled_memory_repre=tf.tile(memory.get_memory_repre(),[self.num_options,1,1])
        self.memory_shape=tf.shape(tiled_memory_repre)
        self.state_shape=tf.shape(matchers[0].aggregation_representation)
        all_probs=[]
        all_states=[]

        cur_states=[]
        for matcher in matchers:
            cur_states.append(matcher.aggregation_representation)



        for step in range(self.num_steps):
            input_vector=self.cal_attention_vector()
            _, new_state=self.cell()


    def map_tensor(self, weight, tensor, vector_dim, target_shape):
        reshaped_tensor=tf.reshape(tensor, [-1,vector_dim])
        mapped_reshaped=tf.matmul(reshaped_tensor,weight)
        return tf.reshape(mapped_reshaped, target_shape)
    def cal_attention_vector(self,cur_state, memory_repre, memory_mask):
        mapped_memory=self.map_tensor(self.W_mem, memory_repre, self.memory_dim, self.memory_shape)
        mapped_state=self.map_tensor(self.W_mem, cur_state, self.state_dim, self.state_shape)
        expanded_state=tf.expand_dims(mapped_state,axis=1)
        relevancy_mat=tf.squeeze(match_utils.cal_relevancy_matrix(expanded_state,mapped_memory))*self.lambda_multiplier
        relevancy_mat=exp_mask(relevancy_mat,memory_mask)
        softmax_sim=tf.nn.softmax(relevancy_mat)
        res = tf.multiply(memory_repre,tf.expand_dims(relevancy_mat,axis=2))
        return tf.reduce_sum(res,axis=1)




