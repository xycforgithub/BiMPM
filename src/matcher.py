import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn
from match_utils import multi_highway_layer
from my_rnn import SwitchableDropoutWrapper
import match_utils


class Matcher:
    def __init__(self,matching_id, question_lengths, choice_lengths, qc_lengths=None, cond_training=True):
        self.question_repre=[]
        self.choice_repre=[]
        self.question_repre_dim=0
        self.choice_repre_dim=0
        self.matching_id=matching_id
        self.question_lengths=question_lengths
        self.choice_lengths=choice_lengths
        self.qc_lengths=qc_lengths
        self.cond_training=cond_training
    def concat(self,is_training,dropout_rate):
        if self.question_repre_dim>0:
            self.question_repre=tf.concat(self.question_repre, 2, name='conat_question_{}'.format(self.matching_id))
            if self.cond_training:
                self.question_repre=match_utils.apply_dropout(self.question_repre, is_training,dropout_rate)
            else:
                if is_training:
                    self.question_repre = tf.nn.dropout(self.question_repre, (1 - dropout_rate))
                else:
                    self.question_repre = tf.multiply(self.question_repre, (1 - dropout_rate))
        if self.choice_repre_dim>0:
            self.choice_repre=tf.concat(self.choice_repre, 2, name='conat_choice_{}'.format(self.matching_id))
            if self.cond_training:
                self.choice_repre=match_utils.apply_dropout(self.choice_repre, is_training,dropout_rate)
            else:
                if is_training:
                    self.choice_repre = tf.nn.dropout(self.choice_repre, (1 - dropout_rate))
                else:
                    self.choice_repre = tf.multiply(self.choice_repre, (1 - dropout_rate))          
    def add_question_repre(self,question_repre,question_dim, extend=False):
        if extend:
            self.question_repre.extend(question_repre)
        else:
            self.question_repre.append(question_repre)
        self.question_repre_dim+=question_dim
    def add_choice_repre(self,choice_repre, choice_dim,extend=False):
        if extend:
            self.choice_repre.extend(choice_repre)
        else:
            self.choice_repre.append(choice_repre)
        self.choice_repre_dim+=choice_dim
    def add_highway_layer(self,highway_layer_num, name,reuse=False):
        if self.question_repre_dim>0:
            with tf.variable_scope("{}_ques".format(name),reuse=reuse):
                self.question_repre = multi_highway_layer(self.question_repre, self.question_repre_dim, highway_layer_num)
        if self.choice_repre_dim>0:
            with tf.variable_scope("{}.choice".format(name),reuse=reuse):
                self.choice_repre = multi_highway_layer(self.choice_repre, self.choice_repre_dim, highway_layer_num)
    def aggregate(self,name,aggregation_layer_num, aggregation_lstm_dim, is_training, dropout_rate, reuse=False):
        self.aggregation_representation = []
        self.aggregation_dim = 0

        aggregation_inputs=[]
        aggregation_dims=[]
        aggregation_lengths=[]

        split_double=False

        if self.question_repre_dim>0:
            aggregation_inputs.append(self.question_repre)
            aggregation_dims.append(self.question_repre_dim)
            if self.qc_lengths is not None:
                aggregation_lengths.append(self.qc_lengths)
            else:
                aggregation_lengths.append(self.question_lengths)

        if self.choice_repre_dim>0:
            aggregation_inputs.append(self.choice_repre)
            aggregation_dims.append(self.choice_repre_dim)
            aggregation_lengths.append(self.choice_lengths)
        else:
            split_double=True

        
        '''
        if with_mean_aggregation:
            self.aggregation_representation.append(tf.reduce_mean(left_repres, axis=1))
            self.aggregation_dim += left_dim
            self.aggregation_representation.append(tf.reduce_mean(right_repres, axis=1))
            self.aggregation_dim += right_dim
        #'''
        with tf.variable_scope('aggregation_layer'):
            for i in range(aggregation_layer_num): # support multiple aggregation layer
                for rep_id in range(len(aggregation_inputs)):
                    with tf.variable_scope('{}-{}-{}'.format(name,rep_id, i), reuse=reuse):
                        aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                        aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                        if self.cond_training:
                            aggregation_lstm_cell_fw=SwitchableDropoutWrapper(aggregation_lstm_cell_fw, is_training, output_keep_prob=(1-dropout_rate))
                        elif is_training:
                            aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                            aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                        aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])
                        cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                                aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_inputs[rep_id], 
                                dtype=tf.float32, sequence_length=aggregation_lengths[rep_id])                      

                        if split_double:
                            fw_rep_q,bw_rep_q,fw_rep_c,bw_rep_c = my_rnn.extract_double_repre(
                                cur_aggregation_representation[0],cur_aggregation_representation[1], self.question_lengths)
                            self.aggregation_representation.append(fw_rep_q)
                            self.aggregation_representation.append(bw_rep_q)
                            self.aggregation_representation.append(fw_rep_c)
                            self.aggregation_representation.append(bw_rep_c)
                            self.aggregation_dim += 4*aggregation_lstm_dim
                        else:
                            fw_rep = cur_aggregation_representation[0][:,-1,:]
                            bw_rep = cur_aggregation_representation[1][:,0,:]
                            self.aggregation_representation.append(fw_rep)
                            self.aggregation_representation.append(bw_rep)
                            self.aggregation_dim += 2* aggregation_lstm_dim
        #
        self.aggregation_representation = tf.concat(self.aggregation_representation, 1) # [batch_size, self.aggregation_dim]
        return self.aggregation_dim
    def add_aggregation_highway(self,highway_layer_num, name, reuse=False):
            # ======Highway layer======
        with tf.variable_scope(name,reuse=reuse):
            agg_shape = tf.shape(self.aggregation_representation)
            batch_size = agg_shape[0]
            self.aggregation_representation = tf.reshape(self.aggregation_representation, [1, batch_size, self.aggregation_dim])
            self.aggregation_representation = multi_highway_layer(self.aggregation_representation, self.aggregation_dim, highway_layer_num)
            self.aggregation_representation = tf.reshape(self.aggregation_representation, [batch_size, self.aggregation_dim])
    def add_softmax_pred(self,w_0,b_0,w_1,b_1, is_training, dropout_rate, use_options=True, num_options=4):
        logits = tf.matmul(self.aggregation_representation, w_0) + b_0
        logits = tf.tanh(logits)
        if self.cond_training:
            logits=match_utils.apply_dropout(logits,is_training,dropout_rate)
        elif is_training:
            logits = tf.nn.dropout(logits, (1 - dropout_rate))
        else:
            logits = tf.multiply(logits, (1 - dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.logits=logits
        if use_options:
            logits=tf.reshape(logits,[-1,num_options])

            self.prob = tf.nn.softmax(logits)
            self.log_prob=tf.nn.log_softmax(logits)


        else:
            self.prob = tf.nn.softmax(logits)
            self.log_prob=tf.nn.log_softmax(logits)

# class qc_Matcher:
#     def __init__():
#         self.qc_repre=[]
#         self.qc_repre_dim=[]
#     def concat():
#         self.qc_repre=tf.concat(self.qc_repre, 2)
#     def add_repre(qc_repre,qc_dim, extend=False):
#         if extend:
#             self.qc_repre.extend(qc_repre)
#         else:
#             self.qc_repre.append(qc_repre)
#         self.qc_repre_dim+=qc_dim