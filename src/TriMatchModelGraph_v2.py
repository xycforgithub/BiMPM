import tensorflow as tf
import my_rnn
import match_utils
from gated_trilateral_match import gated_trilateral_match
from my_rnn import SwitchableDropoutWrapper
from reasonet import ReasoNetModule


class TriMatchModelGraph(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, POS_vocab=None, NER_vocab=None,
                 dropout_rate=0.5, learning_rate=0.001, optimize_type='adam', lambda_l2=1e-5,
                 with_word=True, with_char=True, with_POS=True, with_NER=True,
                 char_lstm_dim=20, context_lstm_dim=100, aggregation_lstm_dim=200, is_training=True,
                 filter_layer_threshold=0.2,
                 MP_dim=50, context_layer_num=1, aggregation_layer_num=1, fix_word_vec=False, with_filter_layer=True,
                 with_highway=False,
                 word_level_MP_dim=-1, sep_endpoint=False, end_model_combine=False, with_match_highway=False,
                 with_aggregation_highway=False, highway_layer_num=1,
                 match_to_passage=True, match_to_question=False, match_to_choice=False, with_no_match=False,
                 with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                 with_max_attentive_match=True, use_options=False,
                 num_options=-1, verbose=False, matching_option=0,
                 concat_context=False, tied_aggre=False, rl_training_method='contrastive', rl_matches=None,
                 cond_training=False, reasonet_training=False, reasonet_steps=5, reasonet_hidden_dim=128,
                 reasonet_lambda=10, reasonet_terminate_mode='original', reasonet_keep_first=False, efficient=False, 
                 reasonet_logit_combine='sum', tied_match=False):
        ''' Matching Options:
        0:a1=q->p, a2=c->p, [concat(a1->a2,a2->a1)]
        1:a1=q->p, a2=c->p, [a1->a2,a2->a1]
        2:[q->p,c->p]
        3:a1=p->q, a2=p->c, [a1->a2,a2->a1]
        4:[q->p,p->q,p->c]
        5:a1=q->p, a2=p->q, a3=p->c,[a3->a1,a3->a2]
        6:[p->q,p->c]
        7: Gated matching
            concat_context: Concat question & choice and feed into context LSTM
            tied_aggre: aggregation layer weights are tied.
            training_method: contrastive reward or policy gradient or soft voting

        RL training method:
        soft_voting: Simple voting training without RL
        contrastive: Basic contrastive reward
        contrastive_imp: Use (r/b-1) instead of (r-b) as in ReasoNet.

        '''
        reasonet_calculated_steps=reasonet_steps+1 if reasonet_keep_first else reasonet_steps

        # ======word representation layer======


        in_question_repres = []
        in_passage_repres = []
        in_choice_repres = []
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.choice_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None])  # [batch_size]
        if cond_training:
            self.is_training = tf.placeholder(tf.bool, [])
        else:
            self.is_training = is_training
        self.concat_idx_mat = None
        self.split_idx_mat_q = None
        self.split_idx_mat_c = None
        if matching_option == 7:
            self.concat_idx_mat = tf.placeholder(tf.int32, [None, None, 2], name='concat_idx_mat')
            if concat_context:
                self.split_idx_mat_q = tf.placeholder(tf.int32, [None, None, 2])
                self.split_idx_mat_c = tf.placeholder(tf.int32, [None, None, 2])
        input_dim = 0
        if with_word and word_vocab is not None:
            self.in_question_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.in_passage_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            self.in_choice_words = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            #             self.word_embedding = tf.get_variable("word_embedding", shape=[word_vocab.size()+1, word_vocab.word_dim], initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            print('!!!shape=', word_vocab.word_vecs.shape)
            with tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                                      initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding,
                                                             self.in_question_words)  # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding,
                                                            self.in_passage_words)  # [batch_size, passage_len, word_dim]
            in_choice_word_repres = tf.nn.embedding_lookup(self.word_embedding,
                                                           self.in_choice_words)  # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)
            in_choice_repres.append(in_choice_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_shape = tf.shape(self.in_choice_words)
            choice_len = input_shape[1]
            input_dim += word_vocab.word_dim

        if with_POS and POS_vocab is not None:
            self.in_question_POSs = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.in_passage_POSs = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            #             self.POS_embedding = tf.get_variable("POS_embedding", shape=[POS_vocab.size()+1, POS_vocab.word_dim], initializer=tf.constant(POS_vocab.word_vecs), dtype=tf.float32)
            self.POS_embedding = tf.get_variable("POS_embedding", initializer=tf.constant(POS_vocab.word_vecs),
                                                 dtype=tf.float32)

            in_question_POS_repres = tf.nn.embedding_lookup(self.POS_embedding,
                                                            self.in_question_POSs)  # [batch_size, question_len, POS_dim]
            in_passage_POS_repres = tf.nn.embedding_lookup(self.POS_embedding,
                                                           self.in_passage_POSs)  # [batch_size, passage_len, POS_dim]
            in_question_repres.append(in_question_POS_repres)
            in_passage_repres.append(in_passage_POS_repres)

            input_shape = tf.shape(self.in_question_POSs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_POSs)
            passage_len = input_shape[1]
            input_dim += POS_vocab.word_dim

        if with_NER and NER_vocab is not None:
            self.in_question_NERs = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.in_passage_NERs = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            #             self.NER_embedding = tf.get_variable("NER_embedding", shape=[NER_vocab.size()+1, NER_vocab.word_dim], initializer=tf.constant(NER_vocab.word_vecs), dtype=tf.float32)
            self.NER_embedding = tf.get_variable("NER_embedding", initializer=tf.constant(NER_vocab.word_vecs),
                                                 dtype=tf.float32)

            in_question_NER_repres = tf.nn.embedding_lookup(self.NER_embedding,
                                                            self.in_question_NERs)  # [batch_size, question_len, NER_dim]
            in_passage_NER_repres = tf.nn.embedding_lookup(self.NER_embedding,
                                                           self.in_passage_NERs)  # [batch_size, passage_len, NER_dim]
            in_question_repres.append(in_question_NER_repres)
            in_passage_repres.append(in_passage_NER_repres)

            input_shape = tf.shape(self.in_question_NERs)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_NERs)
            passage_len = input_shape[1]
            input_dim += NER_vocab.word_dim

        if with_char and char_vocab is not None:
            self.question_char_lengths = tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            self.choice_char_lengths = tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32,
                                                    [None, None, None])  # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32,
                                                   [None, None, None])  # [batch_size, passage_len, p_char_len]
            self.in_choice_chars = tf.placeholder(tf.int32, [None, None, None])  # [batch_size, passage_len, p_char_len]
            input_shape = tf.shape(self.in_question_chars)
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            input_shape = tf.shape(self.in_choice_chars)
            batch_size = input_shape[0]
            choice_len = input_shape[1]
            c_char_len = input_shape[2]

            char_dim = char_vocab.word_dim

            #             self.char_embedding = tf.get_variable("char_embedding", shape=[char_vocab.size()+1, char_vocab.word_dim], initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)
            self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs),
                                                  dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                             self.in_question_chars)  # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                            self.in_passage_chars)  # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            in_choice_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                           self.in_choice_chars)  # [batch_size, passage_len, p_char_len, char_dim]
            in_choice_char_repres = tf.reshape(in_choice_char_repres, shape=[-1, c_char_len, char_dim])
            choice_char_lengths = tf.reshape(self.choice_char_lengths, [-1])

            with tf.variable_scope('char_lstm'):
                # lstm cell
                char_lstm_cell = tf.contrib.rnn.BasicLSTMCell(char_lstm_dim)
                # dropout
                if cond_training:
                    char_lstm_cell = SwitchableDropoutWrapper(char_lstm_cell, self.is_training,
                                                              input_keep_prob=(1 - dropout_rate))
                elif is_training:
                    char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - dropout_rate))

                # if is_training: char_lstm_cell = tf.contrib.rnn.DropoutWrapper(char_lstm_cell, output_keep_prob=(1 - dropout_rate))
                char_lstm_cell = tf.contrib.rnn.MultiRNNCell([char_lstm_cell])

                # question_representation
                question_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_question_char_repres,
                                                           sequence_length=question_char_lengths, dtype=tf.float32)[
                    0]  # [batch_size*question_len, q_char_len, char_lstm_dim]
                question_char_outputs = question_char_outputs[:, -1, :]
                question_char_outputs = tf.reshape(question_char_outputs, [-1, question_len, char_lstm_dim])

                tf.get_variable_scope().reuse_variables()
                # passage representation
                passage_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_passage_char_repres,
                                                          sequence_length=passage_char_lengths, dtype=tf.float32)[
                    0]  # [batch_size*question_len, q_char_len, char_lstm_dim]
                passage_char_outputs = passage_char_outputs[:, -1, :]
                passage_char_outputs = tf.reshape(passage_char_outputs, [-1, passage_len, char_lstm_dim])

                tf.get_variable_scope().reuse_variables()
                # choice representation
                choice_char_outputs = my_rnn.dynamic_rnn(char_lstm_cell, in_choice_char_repres,
                                                         sequence_length=choice_char_lengths, dtype=tf.float32)[
                    0]  # [batch_size*question_len, q_char_len, char_lstm_dim]
                choice_char_outputs = choice_char_outputs[:, -1, :]
                choice_char_outputs = tf.reshape(choice_char_outputs, [-1, choice_len, char_lstm_dim])

            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)
            in_choice_repres.append(choice_char_outputs)

            input_dim += char_lstm_dim

        in_question_repres = tf.concat(in_question_repres, 2)  # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(in_passage_repres, 2)  # [batch_size, passage_len, dim]
        in_choice_repres = tf.concat(in_choice_repres, 2)  # [batch_size, passage_len, dim]

        if cond_training:
            in_question_repres = match_utils.apply_dropout(in_question_repres, self.is_training, dropout_rate)
            in_passage_repres = match_utils.apply_dropout(in_passage_repres, self.is_training, dropout_rate)
            in_choice_repres = match_utils.apply_dropout(in_choice_repres, self.is_training, dropout_rate)
        elif is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))
            in_choice_repres = tf.nn.dropout(in_choice_repres, (1 - dropout_rate))
        else:
            in_question_repres = tf.multiply(in_question_repres, (1 - dropout_rate))
            in_passage_repres = tf.multiply(in_passage_repres, (1 - dropout_rate))
            in_choice_repres = tf.multiply(in_choice_repres, (1 - dropout_rate))

        # if is_training:
        #     in_question_repres = tf.nn.dropout(in_question_repres, (1 - dropout_rate))
        #     in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - dropout_rate))
        #     in_choice_repres = tf.nn.dropout(in_choice_repres, (1 - dropout_rate))
        # else:
        #     in_question_repres = tf.multiply(in_question_repres, (1 - dropout_rate))
        #     in_passage_repres = tf.multiply(in_passage_repres, (1 - dropout_rate))
        #     in_choice_repres = tf.multiply(in_choice_repres, (1 - dropout_rate))


        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32)  # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len,
                                         dtype=tf.float32)  # [batch_size, question_len]
        choice_mask = tf.sequence_mask(self.choice_lengths, choice_len, dtype=tf.float32)  # [batch_size, question_len]

        # ======Highway layer======
        if with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_choice_repres = match_utils.multi_highway_layer(in_choice_repres, input_dim, highway_layer_num)
        # ========Bilateral Matching=====
        # if verbose:
        if matching_option == 7:
            ret_list= gated_trilateral_match(
                in_question_repres, in_passage_repres, in_choice_repres,
                self.question_lengths, self.passage_lengths, self.choice_lengths, question_mask, mask, choice_mask,
                self.concat_idx_mat, self.split_idx_mat_q, self.split_idx_mat_c,
                MP_dim, input_dim, context_layer_num, context_lstm_dim, self.is_training, dropout_rate,
                with_match_highway, aggregation_layer_num, aggregation_lstm_dim, highway_layer_num,
                with_aggregation_highway, with_full_match, with_maxpool_match, with_attentive_match,
                with_max_attentive_match,
                concat_context=concat_context, tied_aggre=tied_aggre, rl_matches=rl_matches,
                cond_training=cond_training,
                efficient=efficient, tied_match=tied_match, construct_memory=reasonet_training, debug=verbose)
            all_match_templates, match_dim, gate_input=ret_list[0:3]
            if verbose:
                self.matching_vectors=ret_list[-1]
                self.matching_vectors.append(gate_input)
            if reasonet_training:
                memory=ret_list[3]
                # tiled_memory_mask=ret_list[4]
        else:
            ret_list= match_utils.trilateral_match(
                in_question_repres, in_passage_repres, in_choice_repres,
                self.question_lengths, self.passage_lengths, self.choice_lengths, question_mask, mask, choice_mask,
                MP_dim, input_dim,
                context_layer_num, context_lstm_dim, self.is_training, dropout_rate,
                with_match_highway, aggregation_layer_num, aggregation_lstm_dim, highway_layer_num,
                with_aggregation_highway,
                with_full_match, with_maxpool_match, with_attentive_match, with_max_attentive_match,
                match_to_passage, match_to_question, match_to_choice, with_no_match, debug=verbose,
                matching_option=matching_option)
            match_representation, match_dim=ret_list[0:2]
            if verbose:
                self.matching_vectors=ret_list[-1]

        print('check: match_dim=', match_dim)
        # ========Prediction Layer=========
        with tf.variable_scope('prediction_layer'):
            w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)

            if use_options:
                w_1 = tf.get_variable("w_1", [match_dim / 2, 1], dtype=tf.float32)
                b_1 = tf.get_variable("b_1", [1], dtype=tf.float32)
            else:
                w_1 = tf.get_variable("w_1", [match_dim / 2, num_classes], dtype=tf.float32)
                b_1 = tf.get_variable("b_1", [num_classes], dtype=tf.float32)

        if matching_option == 7:

            with tf.variable_scope('rl_decision_gate'):
                if use_options and (not efficient):
                    gate_input = gate_input[::num_options, :]
                w_gate = tf.get_variable('w_gate', [2 * context_layer_num * context_lstm_dim, len(rl_matches)], dtype=tf.float32)
                b_gate = tf.get_variable('b_gate', [len(rl_matches)], dtype=tf.float32)
                gate_logits = tf.matmul(gate_input, w_gate) + b_gate

                gate_prob = tf.nn.softmax(gate_logits) # [batch_size/4, num_match]

                gate_log_prob = tf.nn.log_softmax(gate_logits) # [batch_size/4, num_match]

            if not reasonet_training:
                sliced_gate_probs = tf.split(gate_prob, len(rl_matches), axis=1)
                sliced_gate_log_probs = tf.split(gate_log_prob, len(rl_matches), axis=1)
                # if use_options:
                #     tile_times=tf.constant([1,num_options])
                # else:
                #     tile_times=tf.constant([1,num_classes])
                self.gate_prob = gate_prob
                self.gate_log_prob = gate_log_prob
                weighted_probs = []
                weighted_log_probs = []
                all_probs = []
                layout = 'question_first' if efficient else 'choice_first'
                for mid, matcher in enumerate(all_match_templates):
                    matcher.add_softmax_pred(w_0, b_0, w_1, b_1, self.is_training, dropout_rate, use_options, num_options,
                                             layout=layout)
                    all_probs.append(matcher.prob)
                    weighted_probs.append(tf.multiply(matcher.prob, sliced_gate_probs[mid]))
                    weighted_log_probs.append(tf.add(matcher.log_prob, sliced_gate_log_probs[mid]))

                if verbose:
                    self.all_probs = tf.stack(all_probs, axis=0)
                weighted_log_probs = tf.stack(weighted_log_probs, axis=0)
                self.weighted_log_probs = weighted_log_probs
                self.prob = tf.add_n(weighted_probs)
                weighted_probs = tf.stack(weighted_probs, axis=0)
            else:
                self.gate_prob = gate_prob
                self.gate_log_prob = gate_log_prob                # assert efficient
                with tf.variable_scope('reasonet'):
                    reasonet_module=ReasoNetModule(reasonet_steps,num_options,match_dim, memory.aggregation_dim,
                                                   reasonet_hidden_dim, reasonet_lambda, memory_max_len=passage_len, 
                                                   terminate_mode=reasonet_terminate_mode, keep_first=reasonet_keep_first,
                                                   logit_combine=reasonet_logit_combine)
                    all_log_probs, all_states=reasonet_module.multiread_matching(all_match_templates,memory)
                    # [num_steps , num_matchers, batch_size/4], [num_steps * num_matchers * batch_size, state_dim]
                    if verbose:
                        self.matching_vectors.append(all_states)
                        for matcher in all_match_templates:
                            self.matching_vectors.append(matcher.aggregation_representation)

                    # if verbose:
                    #     self.matching_vectors+=reasonet_module.test_vectors
                
                    self.rn_log_probs=all_log_probs
                    num_matcher=len(rl_matches)
                    total_num_gates=num_matcher*reasonet_calculated_steps
                    # all_log_probs=tf.reshape(all_log_probs,[reasonet_calculated_steps, num_matcher,-1]) # [num_steps, num_matcher, batch_size/4]
                    print('gate_log_prob:',gate_log_prob.get_shape())
                    print('all_log_probs:',all_log_probs.get_shape())
                    final_log_probs=tf.reshape(tf.transpose(gate_log_prob)+all_log_probs, [total_num_gates,-1]) #[num_gates, batch_size/4]
                    self.final_log_probs=final_log_probs
                    layout = 'question_first' if efficient else 'choice_first'
                    gate_log_predictions=match_utils.softmax_pred(all_states,w_0,b_0,w_1,b_1,self.is_training,dropout_rate,
                                                                use_options,num_options,cond_training, layout=layout, num_gates=total_num_gates) # [num_gates * batch_size/4, num_options]
                    # gate_log_predictions=tf.reshape(gate_log_predictions, [total_num_gates, -1, num_options]) # [num_gates, batch_size/4, num_options]
                    if verbose:
                        for matcher in all_match_templates:
                            matcher.add_softmax_pred(w_0, b_0, w_1, b_1, self.is_training, dropout_rate, use_options, num_options, layout=layout)    
                            self.matching_vectors.append(matcher.log_prob)

                                                                    
                    if verbose:
                        self.all_probs=gate_log_predictions

                    weighted_log_probs=tf.expand_dims(final_log_probs,axis=2) + gate_log_predictions# [num_gates, batch_size/4, num_options]
                    self.weighted_log_probs = weighted_log_probs
                    weighted_probs=tf.exp(weighted_log_probs)# [num_gates, batch_size/4, num_options]
                    self.prob=tf.reduce_sum(weighted_probs, axis=0)# [batch_size, num_options]
                    print('finished probs')




            if use_options:
                if efficient:
                    gold_matrix = tf.transpose(tf.reshape(self.truth, [num_options, -1]))# [batch_size, num_options]
                else:
                    gold_matrix = tf.reshape(self.truth, [-1, num_options])# [batch_size, num_options]
                gold_matrix = tf.cast(gold_matrix, tf.float32)
                self.gold_matrix=gold_matrix
                correct = tf.equal(tf.argmax(self.prob, 1), tf.argmax(gold_matrix, 1))
            else:
                gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
                #         gold_matrix = tf.one_hot(self.truth, num_classes)

                correct = tf.nn.in_top_k(self.prob, self.truth, 1)
            self.correct = correct
            self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
            self.predictions = tf.arg_max(self.prob, 1)

            if rl_training_method == 'soft_voting':
                self.log_prob = tf.reduce_logsumexp(weighted_log_probs, axis=0)# [batch_size, num_options]
                self.loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(gold_matrix, self.log_prob), axis=1)))
            elif rl_training_method == 'contrastive' or rl_training_method == 'contrastive_imp':

                reward_matrix = gold_matrix# [batch_size, num_options]
                baseline = tf.reduce_sum(tf.multiply(weighted_probs, reward_matrix), axis=[0, 2], keep_dims=True)# [batch_size]
                if rl_training_method == 'contrastive':
                    normalized_reward = reward_matrix - baseline# [batch_size, num_options]
                else:
                    normalized_reward = tf.divide(reward_matrix, baseline) - 1# [batch_size, num_options]
                log_coeffs = tf.multiply(weighted_probs, normalized_reward)
                log_coeffs = tf.stop_gradient(log_coeffs)
                self.log_coeffs = log_coeffs
                self.weighted_log_probs = weighted_log_probs
                self.loss = tf.negative(tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(weighted_log_probs, log_coeffs), axis=[0, 2])))


        else:

            logits = tf.matmul(match_representation, w_0) + b_0
            logits = tf.tanh(logits)

            if cond_training:
                logits = match_utils.apply_dropout(logits, self.is_training, dropout_rate)
            elif is_training:
                logits = tf.nn.dropout(logits, (1 - dropout_rate))
            else:
                logits = tf.multiply(logits, (1 - dropout_rate))
            logits = tf.matmul(logits, w_1) + b_1

            self.final_logits = logits
            if use_options:
                if efficient:
                    logits = tf.transpose(tf.reshape(logits, [num_options, -1]))
                    gold_matrix = tf.transpose(tf.reshape(self.truth, [num_options,-1]))
                else:
                    logits = tf.reshape(logits, [-1, num_options])
                    gold_matrix = tf.reshape(self.truth, [-1, num_options])

                self.prob = tf.nn.softmax(logits)

                #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(self.truth, tf.int64), name='cross_entropy_per_example')
                #         self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

                # gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
                #         gold_matrix = tf.one_hot(self.truth, num_classes)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

                # correct = tf.nn.in_top_k(logits, self.truth, 1)
                # self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
                correct = tf.equal(tf.argmax(logits, 1), tf.argmax(gold_matrix, 1))
                self.gold_matrix=gold_matrix
                self.correct = correct

            else:
                self.prob = tf.nn.softmax(logits)

                #         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(self.truth, tf.int64), name='cross_entropy_per_example')
                #         self.loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

                gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
                #         gold_matrix = tf.one_hot(self.truth, num_classes)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))

                correct = tf.nn.in_top_k(logits, self.truth, 1)
                self.correct = correct
            self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
            self.predictions = tf.arg_max(self.prob, 1)

        if optimize_type == 'adadelta':
            clipper = 50
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
        elif optimize_type == 'sgd':
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)  # Create a variable to track the global step.
            min_lr = 0.000001
            self._lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
        elif optimize_type == 'ema':
            tvars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
            maintain_averages_op = ema.apply(tvars)
            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([train_op]):
                self.train_op = tf.group(maintain_averages_op)
        elif optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(list(zip(grads, tvars)))

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)

        with tf.name_scope('summary'):
            self.loss_summary=tf.summary.scalar('loss',self.loss)
            self.acc_summary = tf.summary.scalar('accuracy',self.eval_correct)

    def get_predictions(self):
        return self.__predictions

    def set_predictions(self, value):
        self.__predictions = value

    def del_predictions(self):
        del self.__predictions

    def get_eval_correct(self):
        return self.__eval_correct

    def set_eval_correct(self, value):
        self.__eval_correct = value

    def del_eval_correct(self):
        del self.__eval_correct

    def get_question_lengths(self):
        return self.__question_lengths

    def get_passage_lengths(self):
        return self.__passage_lengths

    def get_truth(self):
        return self.__truth

    def get_in_question_words(self):
        return self.__in_question_words

    def get_in_passage_words(self):
        return self.__in_passage_words

    def get_word_embedding(self):
        return self.__word_embedding

    def get_in_question_poss(self):
        return self.__in_question_POSs

    def get_in_passage_poss(self):
        return self.__in_passage_POSs

    def get_pos_embedding(self):
        return self.__POS_embedding

    def get_in_question_ners(self):
        return self.__in_question_NERs

    def get_in_passage_ners(self):
        return self.__in_passage_NERs

    def get_ner_embedding(self):
        return self.__NER_embedding

    def get_question_char_lengths(self):
        return self.__question_char_lengths

    def get_passage_char_lengths(self):
        return self.__passage_char_lengths

    def get_in_question_chars(self):
        return self.__in_question_chars

    def get_in_passage_chars(self):
        return self.__in_passage_chars

    def get_char_embedding(self):
        return self.__char_embedding

    def get_prob(self):
        return self.__prob

    def get_prediction(self):
        return self.__prediction

    def get_loss(self):
        return self.__loss

    def get_train_op(self):
        return self.__train_op

    def get_global_step(self):
        return self.__global_step

    def get_lr_rate(self):
        return self.__lr_rate

    def set_question_lengths(self, value):
        self.__question_lengths = value

    def set_passage_lengths(self, value):
        self.__passage_lengths = value

    def set_truth(self, value):
        self.__truth = value

    def set_in_question_words(self, value):
        self.__in_question_words = value

    def set_in_passage_words(self, value):
        self.__in_passage_words = value

    def set_word_embedding(self, value):
        self.__word_embedding = value

    def set_in_question_poss(self, value):
        self.__in_question_POSs = value

    def set_in_passage_poss(self, value):
        self.__in_passage_POSs = value

    def set_pos_embedding(self, value):
        self.__POS_embedding = value

    def set_in_question_ners(self, value):
        self.__in_question_NERs = value

    def set_in_passage_ners(self, value):
        self.__in_passage_NERs = value

    def set_ner_embedding(self, value):
        self.__NER_embedding = value

    def set_question_char_lengths(self, value):
        self.__question_char_lengths = value

    def set_passage_char_lengths(self, value):
        self.__passage_char_lengths = value

    def set_in_question_chars(self, value):
        self.__in_question_chars = value

    def set_in_passage_chars(self, value):
        self.__in_passage_chars = value

    def set_char_embedding(self, value):
        self.__char_embedding = value

    def set_prob(self, value):
        self.__prob = value

    def set_prediction(self, value):
        self.__prediction = value

    def set_loss(self, value):
        self.__loss = value

    def set_train_op(self, value):
        self.__train_op = value

    def set_global_step(self, value):
        self.__global_step = value

    def set_lr_rate(self, value):
        self.__lr_rate = value

    def del_question_lengths(self):
        del self.__question_lengths

    def del_passage_lengths(self):
        del self.__passage_lengths

    def del_truth(self):
        del self.__truth

    def del_in_question_words(self):
        del self.__in_question_words

    def del_in_passage_words(self):
        del self.__in_passage_words

    def del_word_embedding(self):
        del self.__word_embedding

    def del_in_question_poss(self):
        del self.__in_question_POSs

    def del_in_passage_poss(self):
        del self.__in_passage_POSs

    def del_pos_embedding(self):
        del self.__POS_embedding

    def del_in_question_ners(self):
        del self.__in_question_NERs

    def del_in_passage_ners(self):
        del self.__in_passage_NERs

    def del_ner_embedding(self):
        del self.__NER_embedding

    def del_question_char_lengths(self):
        del self.__question_char_lengths

    def del_passage_char_lengths(self):
        del self.__passage_char_lengths

    def del_in_question_chars(self):
        del self.__in_question_chars

    def del_in_passage_chars(self):
        del self.__in_passage_chars

    def del_char_embedding(self):
        del self.__char_embedding

    def del_prob(self):
        del self.__prob

    def del_prediction(self):
        del self.__prediction

    def del_loss(self):
        del self.__loss

    def del_train_op(self):
        del self.__train_op

    def del_global_step(self):
        del self.__global_step

    def del_lr_rate(self):
        del self.__lr_rate

    def get_choice_lengths(self):

        return self.__choice_lengths

    def get_in_choice_words(self):

        return self.__in_choice_words

    def get_in_choice_poss(self):

        return self.__in_choice_POSs

    def get_in_choice_ners(self):

        return self.__in_choice_NERs

    def get_choice_char_lengths(self):

        return self.__choice_char_lengths

    def get_in_choice_chars(self):

        return self.__in_choice_chars

    def set_choice_lengths(self, value):

        self.__choice_lengths = value

    def set_in_choice_words(self, value):

        self.__in_choice_words = value

    def set_in_choice_poss(self, value):

        self.__in_choice_POSs = value

    def set_in_choice_ners(self, value):

        self.__in_choice_NERs = value

    def set_choice_char_lengths(self, value):

        self.__choice_char_lengths = value

    def set_in_choice_chars(self, value):

        self.__in_choice_chars = value

    def del_choice_lengths(self):

        del self.__choice_lengths

    def del_in_choice_words(self):

        del self.__in_choice_words

    def del_in_choice_poss(self):

        del self.__in_choice_POSs

    def del_in_choice_ners(self):

        del self.__in_choice_NERs

    def del_choice_char_lengths(self):

        del self.__choice_char_lengths

    def del_in_choice_chars(self):

        del self.__in_choice_chars

    question_lengths = property(get_question_lengths, set_question_lengths, del_question_lengths,
                                "question_lengths's docstring")
    passage_lengths = property(get_passage_lengths, set_passage_lengths, del_passage_lengths,
                               "passage_lengths's docstring")
    truth = property(get_truth, set_truth, del_truth, "truth's docstring")
    in_question_words = property(get_in_question_words, set_in_question_words, del_in_question_words,
                                 "in_question_words's docstring")
    in_passage_words = property(get_in_passage_words, set_in_passage_words, del_in_passage_words,
                                "in_passage_words's docstring")
    word_embedding = property(get_word_embedding, set_word_embedding, del_word_embedding, "word_embedding's docstring")
    in_question_POSs = property(get_in_question_poss, set_in_question_poss, del_in_question_poss,
                                "in_question_POSs's docstring")
    in_passage_POSs = property(get_in_passage_poss, set_in_passage_poss, del_in_passage_poss,
                               "in_passage_POSs's docstring")
    POS_embedding = property(get_pos_embedding, set_pos_embedding, del_pos_embedding, "POS_embedding's docstring")
    in_question_NERs = property(get_in_question_ners, set_in_question_ners, del_in_question_ners,
                                "in_question_NERs's docstring")
    in_passage_NERs = property(get_in_passage_ners, set_in_passage_ners, del_in_passage_ners,
                               "in_passage_NERs's docstring")
    NER_embedding = property(get_ner_embedding, set_ner_embedding, del_ner_embedding, "NER_embedding's docstring")
    question_char_lengths = property(get_question_char_lengths, set_question_char_lengths, del_question_char_lengths,
                                     "question_char_lengths's docstring")
    passage_char_lengths = property(get_passage_char_lengths, set_passage_char_lengths, del_passage_char_lengths,
                                    "passage_char_lengths's docstring")
    in_question_chars = property(get_in_question_chars, set_in_question_chars, del_in_question_chars,
                                 "in_question_chars's docstring")
    in_passage_chars = property(get_in_passage_chars, set_in_passage_chars, del_in_passage_chars,
                                "in_passage_chars's docstring")
    char_embedding = property(get_char_embedding, set_char_embedding, del_char_embedding, "char_embedding's docstring")
    prob = property(get_prob, set_prob, del_prob, "prob's docstring")
    prediction = property(get_prediction, set_prediction, del_prediction, "prediction's docstring")
    loss = property(get_loss, set_loss, del_loss, "loss's docstring")
    train_op = property(get_train_op, set_train_op, del_train_op, "train_op's docstring")
    global_step = property(get_global_step, set_global_step, del_global_step, "global_step's docstring")
    lr_rate = property(get_lr_rate, set_lr_rate, del_lr_rate, "lr_rate's docstring")
    eval_correct = property(get_eval_correct, set_eval_correct, del_eval_correct, "eval_correct's docstring")
    predictions = property(get_predictions, set_predictions, del_predictions, "predictions's docstring")

    choice_lengths = property(get_choice_lengths, set_choice_lengths, del_choice_lengths, "choice_lengths's docstring")
    in_choice_words = property(get_in_choice_words, set_in_choice_words, del_in_choice_words,
                               "in_choice_words's docstring")
    in_choice_POSs = property(get_in_choice_poss, set_in_choice_poss, del_in_choice_poss, "in_choice_POSs's docstring")
    in_choice_NERs = property(get_in_choice_ners, set_in_choice_ners, del_in_choice_ners, "in_choice_NERs's docstring")
    choice_char_lengths = property(get_choice_char_lengths, set_choice_char_lengths, del_choice_char_lengths,
                                   "choice_char_lengths's docstring")
    in_choice_chars = property(get_in_choice_chars, set_in_choice_chars, del_in_choice_chars,
                               "in_choice_chars's docstring")
