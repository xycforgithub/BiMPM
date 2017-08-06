import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn
from match_utils import multi_highway_layer
from my_rnn import SwitchableDropoutWrapper
import match_utils
from matcher import Matcher

class Memory(Matcher):
    def __init__(self,memory_lengths, tiled_memory_mask, cond_training=True):
        super().__init__(-1,memory_lengths,None,cond_training=cond_training)
        self.tiled_memory_mask=tiled_memory_mask
    def add_memory_repre(self, memory_repre, memory_dim, extend=False):
        super().add_question_repre(memory_repre,memory_dim,extend)
    def get_memory_repre(self):
        return self.aggregation_representation
    def get_memory_length(self):
        return self.question_lengths
    def get_memory_dim(self):
        return self.aggregation_dim
    def aggregate(self, aggregation_layer_num, aggregation_lstm_dim, is_training, dropout_rate, tied_aggre=False,
                  reuse=None):
        self.aggregation_representation = []
        self.aggregation_dim = 0

        aggregation_inputs = [self.question_repre]
        aggregation_lengths = [self.question_lengths]

        '''
        if with_mean_aggregation:
            self.aggregation_representation.append(tf.reduce_mean(left_repres, axis=1))
            self.aggregation_dim += left_dim
            self.aggregation_representation.append(tf.reduce_mean(right_repres, axis=1))
            self.aggregation_dim += right_dim
        #'''
        with tf.variable_scope('aggregation_layer'):
            for i in range(aggregation_layer_num):  # support multiple aggregation layer
                for rep_id in range(len(aggregation_inputs)):
                    if tied_aggre:
                        name = 'aggre_layer{}'.format(i)
                        if rep_id > 0:
                            reuse = True
                    else:
                        name = 'aggre_layer{}_matcher{}_part{}'.format(i, self.matching_id, rep_id)
                    with tf.variable_scope(name, reuse=reuse):
                        aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim, reuse=reuse)
                        aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim, reuse=reuse)
                        if self.cond_training:
                            aggregation_lstm_cell_fw = SwitchableDropoutWrapper(aggregation_lstm_cell_fw, is_training,
                                                                                output_keep_prob=(1 - dropout_rate))
                            aggregation_lstm_cell_bw = SwitchableDropoutWrapper(aggregation_lstm_cell_bw, is_training,
                                                                                output_keep_prob=(1 - dropout_rate))
                        elif is_training:
                            aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw,
                                                                                     output_keep_prob=(
                                                                                     1 - dropout_rate))
                            aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw,
                                                                                     output_keep_prob=(
                                                                                     1 - dropout_rate))
                        aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                        aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])
                        cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                            aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_inputs[rep_id],
                            dtype=tf.float32, sequence_length=aggregation_lengths[rep_id])
                        self.aggregation_dim += 2 * aggregation_lstm_dim
                        aggregation_inputs[rep_id] = tf.concat(cur_aggregation_representation, 2)
                        self.aggregation_representation.extend(cur_aggregation_representation)
        #
        self.aggregation_representation = tf.concat(self.aggregation_representation,2)  # [batch_size, memory_length, self.aggregation_dim]
        # self.
        return self.aggregation_dim