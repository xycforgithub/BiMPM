import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn
# from tensorflow import DropoutWrapper
# from matcher import Matcher


eps = 1e-6


def softmax_pred(all_states, w_0, b_0, w_1, b_1, is_training, dropout_rate, use_options=True, num_options=4,
                     cond_training=True, layout='choice_first'):
    # Layout = choice_first or question_first
    logits = tf.matmul(all_states, w_0) + b_0
    logits = tf.tanh(logits)
    if cond_training:
        logits = apply_dropout(logits, is_training, dropout_rate)
    elif is_training:
        logits = tf.nn.dropout(logits, (1 - dropout_rate))
    else:
        logits = tf.multiply(logits, (1 - dropout_rate))
    logits = tf.matmul(logits, w_1) + b_1

    if use_options:
        if layout == 'choice_first':
            logits = tf.reshape(logits, [-1, num_options])
        else:
            logits = tf.transpose(tf.reshape(logits, [num_options, -1]))

        # prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)


    else:
        # prob = tf.nn.softmax(logits)
        log_prob = tf.nn.log_softmax(logits)
    return log_prob


def apply_dropout(repre, is_training, dropout_rate):
    return tf.cond(is_training,lambda: tf.nn.dropout(repre, 1 - dropout_rate), lambda:repre)

def cosine_distance(y1,y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
#     cosine_numerator = T.sum(y1*y2, axis=-1)
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
#     y1_norm = T.sqrt(T.maximum(T.sum(T.sqr(y1), axis=-1), eps)) #be careful while using T.sqrt(), like in the cases of Euclidean distance, cosine similarity, for the gradient of T.sqrt() at 0 is undefined, we should add an Eps or use T.maximum(original, eps) in the sqrt.
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps)) 
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps)) 
    return cosine_numerator / y1_norm / y2_norm

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def cal_cosine_weighted_question_representation(question_representation, cosine_matrix, normalize=False):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    if normalize: cosine_matrix = tf.nn.softmax(cosine_matrix)
    expanded_cosine_matrix = tf.expand_dims(cosine_matrix, axis=-1) # [batch_size, passage_len, question_len, 'x']
    weighted_question_words = tf.expand_dims(question_representation, axis=1) # [batch_size, 'x', question_len, dim]
    weighted_question_words = tf.reduce_sum(tf.multiply(weighted_question_words, expanded_cosine_matrix), axis=2)# [batch_size, passage_len, dim]
    if not normalize:
        weighted_question_words = tf.div(weighted_question_words, tf.expand_dims(tf.add(tf.reduce_sum(cosine_matrix, axis=-1),eps),axis=-1))
    return weighted_question_words # [batch_size, passage_len, dim]

def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=2) #[batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0), axis=0) # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)#[batch_size, passage_len, decompse_dim, dim]

def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1) #[batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0) # [1, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params) # [batch_size, decompse_dim, dim]

def multi_perspective_expand_for_1D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=0) #['x', dim]
    return tf.multiply(in_tensor, decompose_params) # [decompse_dim, dim]


def cal_full_matching_bak(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    mp_passage_rep = multi_perspective_expand_for_3D(passage_representation, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    mp_full_question_rep = multi_perspective_expand_for_2D(full_question_representation, decompose_params) # [batch_size, decompse_dim, dim]
    return cosine_distance(mp_passage_rep, tf.expand_dims(mp_full_question_rep, axis=1)) #[batch_size, passage_len, decompse_dim]

def cal_full_matching(passage_representation, full_question_representation, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # full_question_representation: [batch_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_1D(q, decompose_params) # [decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, decompose]
    elems = (passage_representation, full_question_representation)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]
    
def cal_maxpooling_matching_bak(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    passage_rep = multi_perspective_expand_for_3D(passage_rep, decompose_params) # [batch_size, passage_len, decompse_dim, dim]
    question_rep = multi_perspective_expand_for_3D(question_rep, decompose_params) # [batch_size, question_len, decompse_dim, dim]

    passage_rep = tf.expand_dims(passage_rep, 2) # [batch_size, passage_len, 1, decompse_dim, dim]
    question_rep = tf.expand_dims(question_rep, 1) # [batch_size, 1, question_len, decompse_dim, dim]
    matching_matrix = cosine_distance(passage_rep,question_rep) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat([tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)], 2)# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1) # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0) # [1, question_len, decompose_dim, dim]
        return cosine_distance(p, q) # [passage_len, question_len, decompose]
    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat([tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)], 2)# [batch_size, passage_len, 2*decompse_dim]

def cal_maxpooling_matching_for_word(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]
    
    def singel_instance(x):
        p = x[0]
        q = x[1]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [question_len, decompose_dim, dim]
        # p: [pasasge_len, dim], q: [question_len, dim]
        def single_instance_2(y):
            # y: [dim]
            y = multi_perspective_expand_for_1D(y, decompose_params) #[decompose_dim, dim]
            y = tf.expand_dims(y, 0) # [1, decompose_dim, dim]
            matching_matrix = cosine_distance(y, q)#[question_len, decompose_dim]
            return tf.concat([tf.reduce_max(matching_matrix, axis=0), tf.reduce_mean(matching_matrix, axis=0)], 0) #[2*decompose_dim]
        return tf.map_fn(single_instance_2, p, dtype=tf.float32) # [passage_len, 2*decompse_dim]
    elems = (passage_rep, question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, 2*decompse_dim]


def cal_attentive_matching(passage_rep, att_question_rep, decompose_params):
    # passage_rep: [batch_size, passage_len, dim]
    # att_question_rep: [batch_size, passage_len, dim]
    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [pasasge_len, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params) # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params) # [pasasge_len, decompose_dim, dim]
        return cosine_distance(p, q) # [pasasge_len, decompose_dim]

    elems = (passage_rep, att_question_rep)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, decompse_dim]

def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

#     xdev = x - x.max()
#     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
#     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions), mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]
    
def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def cal_max_question_representation(question_representation, cosine_matrix):
    # question_representation: [batch_size, question_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    question_index = tf.arg_max(cosine_matrix, 2) # [batch_size, passage_len]
    def singel_instance(x):
        q = x[0]
        c = x[1]
        return tf.gather(q, c)
    elems = (question_representation, question_index)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, passage_len, dim]

def cal_linear_decomposition_representation(passage_representation, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate):
    # passage_representation: [batch_size, passage_len, dim]
    # cosine_matrix: [batch_size, passage_len, question_len]
    passage_similarity = tf.reduce_max(cosine_matrix, 2)# [batch_size, passage_len]
    similar_weights = tf.expand_dims(passage_similarity, -1) # [batch_size, passage_len, 1]
    dissimilar_weights = tf.subtract(1.0, similar_weights)
    similar_component = tf.multiply(passage_representation, similar_weights)
    dissimilar_component = tf.multiply(passage_representation, dissimilar_weights)
    all_component = tf.concat([similar_component, dissimilar_component], 2)
    if lex_decompsition_dim==-1:
        return all_component
    with tf.variable_scope('lex_decomposition'):
        lex_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(lex_decompsition_dim)
        lex_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(lex_decompsition_dim)
        if is_training:
            lex_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lex_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            lex_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lex_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        lex_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lex_lstm_cell_fw])
        lex_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lex_lstm_cell_bw])

        (lex_features_fw, lex_features_bw), _ = rnn.bidirectional_dynamic_rnn(
                    lex_lstm_cell_fw, lex_lstm_cell_bw, all_component, dtype=tf.float32, sequence_length=passage_lengths)

        lex_features = tf.concat([lex_features_fw, lex_features_bw], 2)
    return lex_features

def match_passage_with_question_direct(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                                with_direction=False):

    fw_question_aware_representatins = []
    bw_question_aware_representatins = []
    fw_dim = 0
    bw_dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        fw_question_full_rep = question_context_representation_fw[:,-1,:]
        bw_question_full_rep = question_context_representation_bw[:,0,:]

        question_context_representation_fw = tf.multiply(question_context_representation_fw, tf.expand_dims(question_mask,-1))
        question_context_representation_bw = tf.multiply(question_context_representation_bw, tf.expand_dims(question_mask,-1))
        passage_context_representation_fw = tf.multiply(passage_context_representation_fw, tf.expand_dims(mask,-1))
        passage_context_representation_bw = tf.multiply(passage_context_representation_bw, tf.expand_dims(mask,-1))

        forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)

        backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)
        if MP_dim > 0:
            if with_full_match:
                # forward Full-Matching: passage_context_representation_fw vs question_context_representation_fw[-1]
                fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_full_match_rep = cal_full_matching(passage_context_representation_fw, fw_question_full_rep, fw_full_decomp_params)
                fw_question_aware_representatins.append(fw_full_match_rep)
                fw_dim += MP_dim

                # backward Full-Matching: passage_context_representation_bw vs question_context_representation_bw[0]
                bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_full_match_rep = cal_full_matching(passage_context_representation_bw, bw_question_full_rep, bw_full_decomp_params)
                bw_question_aware_representatins.append(bw_full_match_rep)
                bw_dim += MP_dim

            if with_maxpool_match:
                # forward Maxpooling-Matching
                fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_fw, question_context_representation_fw, fw_maxpooling_decomp_params)
                fw_question_aware_representatins.append(fw_maxpooling_rep)
                fw_dim += 2*MP_dim
                # backward Maxpooling-Matching
                bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_bw, question_context_representation_bw, bw_maxpooling_decomp_params)
                bw_question_aware_representatins.append(bw_maxpooling_rep)
                bw_dim += 2*MP_dim
            
            if with_attentive_match:
                # forward attentive-matching
                # forward weighted question representation: [batch_size, question_len, passage_len] [batch_size, question_len, context_lstm_dim]
                att_question_fw_contexts = cal_cosine_weighted_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching(passage_context_representation_fw, att_question_fw_contexts, fw_attentive_decomp_params)
                fw_question_aware_representatins.append(fw_attentive_rep)
                fw_dim += MP_dim

                # backward attentive-matching
                # backward weighted question representation
                att_question_bw_contexts = cal_cosine_weighted_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching(passage_context_representation_bw, att_question_bw_contexts, bw_attentive_decomp_params)
                bw_question_aware_representatins.append(bw_attentive_rep)
                bw_dim += MP_dim
            
            if with_max_attentive_match:
                # forward max attentive-matching
                max_att_fw = cal_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_max_attentive_rep = cal_attentive_matching(passage_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
                fw_question_aware_representatins.append(fw_max_attentive_rep)
                fw_dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching(passage_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
                bw_question_aware_representatins.append(bw_max_attentive_rep)
                bw_dim += MP_dim

        fw_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        fw_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        bw_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        bw_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        fw_dim += 2
        bw_dim += 2
    if with_direction:
        return (fw_question_aware_representatins,bw_question_aware_representatins,fw_dim,bw_dim)
    else:
        return (fw_question_aware_representatins+bw_question_aware_representatins,fw_dim+bw_dim)

def match_passage_with_question(passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True):

    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        fw_question_full_rep = question_context_representation_fw[:,-1,:]
        bw_question_full_rep = question_context_representation_bw[:,0,:]

        question_context_representation_fw = tf.multiply(question_context_representation_fw, tf.expand_dims(question_mask,-1))
        question_context_representation_bw = tf.multiply(question_context_representation_bw, tf.expand_dims(question_mask,-1))
        passage_context_representation_fw = tf.multiply(passage_context_representation_fw, tf.expand_dims(mask,-1))
        passage_context_representation_bw = tf.multiply(passage_context_representation_bw, tf.expand_dims(mask,-1))

        forward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_fw, passage_context_representation_fw)
        forward_relevancy_matrix = mask_relevancy_matrix(forward_relevancy_matrix, question_mask, mask)

        backward_relevancy_matrix = cal_relevancy_matrix(question_context_representation_bw, passage_context_representation_bw)
        backward_relevancy_matrix = mask_relevancy_matrix(backward_relevancy_matrix, question_mask, mask)
        if MP_dim > 0:
            if with_full_match:
                # forward Full-Matching: passage_context_representation_fw vs question_context_representation_fw[-1]
                fw_full_decomp_params = tf.get_variable("forward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_full_match_rep = cal_full_matching(passage_context_representation_fw, fw_question_full_rep, fw_full_decomp_params)
                all_question_aware_representatins.append(fw_full_match_rep)
                dim += MP_dim

                # backward Full-Matching: passage_context_representation_bw vs question_context_representation_bw[0]
                bw_full_decomp_params = tf.get_variable("backward_full_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_full_match_rep = cal_full_matching(passage_context_representation_bw, bw_question_full_rep, bw_full_decomp_params)
                all_question_aware_representatins.append(bw_full_match_rep)
                dim += MP_dim

            if with_maxpool_match:
                # forward Maxpooling-Matching
                fw_maxpooling_decomp_params = tf.get_variable("forward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_fw, question_context_representation_fw, fw_maxpooling_decomp_params)
                all_question_aware_representatins.append(fw_maxpooling_rep)
                dim += 2*MP_dim
                # backward Maxpooling-Matching
                bw_maxpooling_decomp_params = tf.get_variable("backward_maxpooling_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_maxpooling_rep = cal_maxpooling_matching(passage_context_representation_bw, question_context_representation_bw, bw_maxpooling_decomp_params)
                all_question_aware_representatins.append(bw_maxpooling_rep)
                dim += 2*MP_dim
            
            if with_attentive_match:
                # forward attentive-matching
                # forward weighted question representation: [batch_size, question_len, passage_len] [batch_size, question_len, context_lstm_dim]
                att_question_fw_contexts = cal_cosine_weighted_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_attentive_decomp_params = tf.get_variable("forward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_attentive_rep = cal_attentive_matching(passage_context_representation_fw, att_question_fw_contexts, fw_attentive_decomp_params)
                all_question_aware_representatins.append(fw_attentive_rep)
                dim += MP_dim

                # backward attentive-matching
                # backward weighted question representation
                att_question_bw_contexts = cal_cosine_weighted_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_attentive_decomp_params = tf.get_variable("backward_attentive_matching_decomp", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_attentive_rep = cal_attentive_matching(passage_context_representation_bw, att_question_bw_contexts, bw_attentive_decomp_params)
                all_question_aware_representatins.append(bw_attentive_rep)
                dim += MP_dim
            
            if with_max_attentive_match:
                # forward max attentive-matching
                max_att_fw = cal_max_question_representation(question_context_representation_fw, forward_relevancy_matrix)
                fw_max_att_decomp_params = tf.get_variable("fw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                fw_max_attentive_rep = cal_attentive_matching(passage_context_representation_fw, max_att_fw, fw_max_att_decomp_params)
                all_question_aware_representatins.append(fw_max_attentive_rep)
                dim += MP_dim

                # backward max attentive-matching
                max_att_bw = cal_max_question_representation(question_context_representation_bw, backward_relevancy_matrix)
                bw_max_att_decomp_params = tf.get_variable("bw_max_att_decomp_params", shape=[MP_dim, context_lstm_dim], dtype=tf.float32)
                bw_max_attentive_rep = cal_attentive_matching(passage_context_representation_bw, max_att_bw, bw_max_att_decomp_params)
                all_question_aware_representatins.append(bw_max_attentive_rep)
                dim += MP_dim

        all_question_aware_representatins.append(tf.reduce_max(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(forward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_max(backward_relevancy_matrix, axis=2,keep_dims=True))
        all_question_aware_representatins.append(tf.reduce_mean(backward_relevancy_matrix, axis=2,keep_dims=True))
        dim += 4
    return (all_question_aware_representatins, dim)
        
def unidirectional_matching(in_question_repres, in_passage_repres,question_lengths, passage_lengths,
                            question_mask, mask, MP_dim, input_dim, with_filter_layer, context_layer_num,
                            context_lstm_dim,is_training,dropout_rate,with_match_highway,aggregation_layer_num,
                            aggregation_lstm_dim,highway_layer_num,with_aggregation_highway,with_lex_decomposition, lex_decompsition_dim,
                            with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True, verbose=False):
    # ======Filter layer======
    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres)
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    raw_in_passage_repres = in_passage_repres
    if with_filter_layer:
        relevancy_matrix = cosine_matrix # [batch_size, passage_len, question_len]
        relevancy_degrees = tf.reduce_max(relevancy_matrix, axis=2) # [batch_size, passage_len]
        relevancy_degrees = tf.expand_dims(relevancy_degrees,axis=-1) # [batch_size, passage_len, 'x']
        in_passage_repres = tf.multiply(in_passage_repres, relevancy_degrees)
        
    # =======Context Representation Layer & Multi-Perspective matching layer=====
    all_question_aware_representatins = []
    # max and mean pooling at word level
    all_question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True))
    all_question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))
    question_aware_dim = 2
    
    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            max_att = cal_max_question_representation(in_question_repres, cosine_matrix)
            max_att_decomp_params = tf.get_variable("max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            max_attentive_rep = cal_attentive_matching(raw_in_passage_repres, max_att, max_att_decomp_params)
            all_question_aware_representatins.append(max_attentive_rep)
            question_aware_dim += MP_dim
    
    # lex decomposition
    if with_lex_decomposition:
        lex_decomposition = cal_linear_decomposition_representation(raw_in_passage_repres, passage_lengths, cosine_matrix,is_training, 
                                            lex_decompsition_dim, dropout_rate)
        all_question_aware_representatins.append(lex_decomposition)
        if lex_decompsition_dim== -1: question_aware_dim += 2 * input_dim
        else: question_aware_dim += 2* lex_decompsition_dim
        
    with tf.variable_scope('context_MP_matching'):
        for i in range(context_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)
                    
                # Multi-perspective matching
                with tf.variable_scope('MP_matching'):
                    (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw,question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    all_question_aware_representatins.extend(matching_vectors)
                    question_aware_dim += matching_dim
        
    all_question_aware_representatins = tf.concat(all_question_aware_representatins, 2) # [batch_size, passage_len, dim]

    if is_training:
        all_question_aware_representatins = tf.nn.dropout(all_question_aware_representatins, (1 - dropout_rate))
    else:
        all_question_aware_representatins = tf.multiply(all_question_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("matching_highway"):
            all_question_aware_representatins = multi_highway_layer(all_question_aware_representatins, question_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    aggregation_input = all_question_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in range(aggregation_layer_num):
            with tf.variable_scope('layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input, 
                        dtype=tf.float32, sequence_length=passage_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, passage_len, 2*aggregation_lstm_dim]
        
    #
    aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    if verbose:
        return (aggregation_representation, aggregation_dim, all_question_aware_representatins)
    return (aggregation_representation, aggregation_dim)
        
def bilateral_match_func1(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True, verbose=False):
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    match_representation = []
    match_dim = 0
        
    reuse_match_params = None
    if with_left_match:
        reuse_match_params = True
        with tf.name_scope("match_passsage"):
            with tf.variable_scope("MP-Match", reuse=None, initializer=initializer):
                ret_list = unidirectional_matching(in_question_repres, in_passage_repres,
                            question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match, verbose=verbose)
                if verbose:
                    (passage_match_representation, passage_match_dim, all_repre) = ret_list
                else:
                    (passage_match_representation, passage_match_dim) = ret_list
                match_representation.append(passage_match_representation)
                match_dim += passage_match_dim
    if with_right_match:
        with tf.name_scope("match_question"):
            with tf.variable_scope("MP-Match", reuse=reuse_match_params, initializer=initializer):
                ret_list = unidirectional_matching(in_passage_repres, in_question_repres, 
                            passage_lengths, question_lengths, mask, question_mask, MP_dim, input_dim, 
                            with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                            with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                            with_aggregation_highway, with_lex_decomposition,lex_decompsition_dim,
                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                            with_attentive_match=with_attentive_match,
                            with_max_attentive_match=with_max_attentive_match, verbose=verbose)
                if verbose:
                    (question_match_representation, question_match_dim, all_repre) = ret_list
                else:
                    (question_match_representation, question_match_dim) = ret_list
                match_representation.append(question_match_representation)
                match_dim += question_match_dim
    match_representation = tf.concat(match_representation, 1)
    if verbose:
        return (match_representation, match_dim, all_repre)
    else:
        return (match_representation, match_dim)



def bilateral_match_func2(in_question_repres, in_passage_repres,
                        question_lengths, passage_lengths, question_mask, mask, MP_dim, input_dim, 
                        with_filter_layer, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway,with_lex_decomposition,lex_decompsition_dim,
                        with_full_match=True, with_maxpool_match=True, with_attentive_match=True, with_max_attentive_match=True,
                        with_left_match=True, with_right_match=True, with_mean_aggregation=True, with_no_match=False):


    cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    cosine_matrix = mask_relevancy_matrix(cosine_matrix, question_mask, mask)
    cosine_matrix_transpose = tf.transpose(cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

    # ====word level matching======
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # max and mean pooling at word level
    question_aware_representatins.append(tf.reduce_max(cosine_matrix, axis=2,keep_dims=True)) # [batch_size, passage_length, 1]
    question_aware_representatins.append(tf.reduce_mean(cosine_matrix, axis=2,keep_dims=True))# [batch_size, passage_length, 1]
    question_aware_dim += 2
    passage_aware_representatins.append(tf.reduce_max(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    passage_aware_representatins.append(tf.reduce_mean(cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
    passage_aware_dim += 2
    

    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            qa_max_att = cal_max_question_representation(in_question_repres, cosine_matrix)# [batch_size, passage_len, dim]
            qa_max_att_decomp_params = tf.get_variable("qa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            qa_max_attentive_rep = cal_attentive_matching(in_passage_repres, qa_max_att, qa_max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
            question_aware_representatins.append(qa_max_attentive_rep)
            question_aware_dim += MP_dim

            pa_max_att = cal_max_question_representation(in_passage_repres, cosine_matrix_transpose)# [batch_size, question_len, dim]
            pa_max_att_decomp_params = tf.get_variable("pa_word_max_att_decomp_params", shape=[MP_dim, input_dim], dtype=tf.float32)
            pa_max_attentive_rep = cal_attentive_matching(in_question_repres, pa_max_att, pa_max_att_decomp_params)# [batch_size, question_len, decompse_dim]
            passage_aware_representatins.append(pa_max_attentive_rep)
            passage_aware_dim += MP_dim

    with tf.variable_scope('context_MP_matching'):
        for i in range(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)
                if with_no_match:
                    print('No match!!!!')
                    question_aware_representatins.append(passage_context_representation_fw)

                    question_aware_representatins.append(passage_context_representation_bw)
                    passage_aware_representatins.append(question_context_representation_fw)
                    passage_aware_representatins.append(question_context_representation_bw)

                    # question_aware_representatins.append(question_context_representation_fw)
                    # print(question_context_representation_fw.shape)
                    # print(question_context_representation_bw.shape)
                    # question_aware_representatins.append(question_context_representation_bw)
                    # passage_aware_representatins.append(passage_context_representation_fw)
                    # passage_aware_representatins.append(passage_context_representation_bw)
                else:
                    # Multi-perspective matching
                    with tf.variable_scope('left_MP_matching'):
                        (matching_vectors, matching_dim) = match_passage_with_question(passage_context_representation_fw, 
                                    passage_context_representation_bw, mask,
                                    question_context_representation_fw, question_context_representation_bw,question_mask,
                                    MP_dim, context_lstm_dim, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                        question_aware_representatins.extend(matching_vectors)
                        question_aware_dim += matching_dim
                    
                    with tf.variable_scope('right_MP_matching'):
                        (matching_vectors, matching_dim) = match_passage_with_question(question_context_representation_fw, 
                                    question_context_representation_bw, question_mask,
                                    passage_context_representation_fw, passage_context_representation_bw,mask,
                                    MP_dim, context_lstm_dim, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                        passage_aware_representatins.extend(matching_vectors)
                        passage_aware_dim += matching_dim
        

        
    question_aware_representatins = tf.concat(question_aware_representatins, 2) # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(passage_aware_representatins, 2) # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.nn.dropout(passage_aware_representatins, (1 - dropout_rate))
    else:
        question_aware_representatins = tf.multiply(question_aware_representatins, (1 - dropout_rate))
        passage_aware_representatins = tf.multiply(passage_aware_representatins, (1 - dropout_rate))
        
    # ======Highway layer======
    if with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(question_aware_representatins, question_aware_dim,highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(passage_aware_representatins, passage_aware_dim,highway_layer_num)
        
    #========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0
    
    '''
    if with_mean_aggregation:
        aggregation_representation.append(tf.reduce_mean(question_aware_representatins, axis=1))
        aggregation_dim += question_aware_dim
        aggregation_representation.append(tf.reduce_mean(passage_aware_representatins, axis=1))
        aggregation_dim += passage_aware_dim
    #'''

    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in range(aggregation_layer_num): # support multiple aggregation layer
            with tf.variable_scope('left_layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, qa_aggregation_input, 
                        dtype=tf.float32, sequence_length=passage_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                qa_aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, passage_len, 2*aggregation_lstm_dim]

            with tf.variable_scope('right_layer-{}'.format(i)):
                aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                if is_training:
                    aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                        aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, pa_aggregation_input, 
                        dtype=tf.float32, sequence_length=question_lengths)

                fw_rep = cur_aggregation_representation[0][:,-1,:]
                bw_rep = cur_aggregation_representation[1][:,0,:]
                aggregation_representation.append(fw_rep)
                aggregation_representation.append(bw_rep)
                aggregation_dim += 2* aggregation_lstm_dim
                pa_aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, passage_len, 2*aggregation_lstm_dim]
    #
    aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

    # ======Highway layer======
    if with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
            aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
    
    return (aggregation_representation, aggregation_dim)

def trilateral_match(in_question_repres, in_passage_repres, in_choice_repres,
                        question_lengths, passage_lengths, choice_lengths, question_mask, mask, choice_mask, MP_dim, input_dim, 
                        context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway, with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                        with_max_attentive_match=True, match_to_passage=True, match_to_question=True, match_to_choice=True, with_no_match=False,
                        debug=False,matching_option=0):

    print('trilateral match ',matching_option)
    qp_cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    qp_cosine_matrix = mask_relevancy_matrix(qp_cosine_matrix, question_mask, mask)
    qp_cosine_matrix_transpose = tf.transpose(qp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

    cp_cosine_matrix = cal_relevancy_matrix(in_choice_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    cp_cosine_matrix = mask_relevancy_matrix(cp_cosine_matrix, choice_mask, mask)
    cp_cosine_matrix_transpose = tf.transpose(cp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]


    # ====word level matching======
    if matching_option==0:
        if match_to_passage:
            matched_passage_representations = []
            matched_passage_dim = 0
        if match_to_question:
            matched_question_representations = []
            matched_question_dim = 0
        if match_to_choice:
            matched_choice_representations = []
            matched_choice_dim = 0
        match_to_passage=True
        match_to_question=False
        match_to_choice=False

        # max and mean pooling at word level
        if match_to_passage:
            matched_passage_representations.append(tf.reduce_max(qp_cosine_matrix, axis=2,keep_dims=True)) # [batch_size, passage_length, 1]
            matched_passage_representations.append(tf.reduce_mean(qp_cosine_matrix, axis=2,keep_dims=True))# [batch_size, passage_length, 1]
            matched_passage_representations.append(tf.reduce_max(cp_cosine_matrix, axis=2,keep_dims=True)) # [batch_size, passage_length, 1]
            matched_passage_representations.append(tf.reduce_mean(cp_cosine_matrix, axis=2,keep_dims=True))# [batch_size, passage_length, 1]
            matched_passage_dim += 4
        if match_to_question:
            matched_question_representations.append(tf.reduce_max(qp_cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
            matched_question_representations.append(tf.reduce_mean(qp_cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
            matched_question_dim += 2
        if match_to_choice:
            matched_choice_representations.append(tf.reduce_max(cp_cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
            matched_choice_representations.append(tf.reduce_mean(cp_cosine_matrix_transpose, axis=2,keep_dims=True))# [batch_size, question_len, 1]
            matched_choice_dim += 2    
    elif matching_option in [1,2]:
        qp_matched_repre=[]
        cp_matched_repre=[]
        matched_qp_dim=0
        matched_cp_dim=0
        qp_matched_repre.append(tf.reduce_max(qp_cosine_matrix, axis=2,keep_dims=True))
        qp_matched_repre.append(tf.reduce_mean(qp_cosine_matrix, axis=2,keep_dims=True))
        cp_matched_repre.append(tf.reduce_max(cp_cosine_matrix, axis=2,keep_dims=True))
        cp_matched_repre.append(tf.reduce_mean(cp_cosine_matrix, axis=2,keep_dims=True))
        matched_cp_dim+=2
        matched_qp_dim+=2
    elif matching_option==3:
        qp_matched_repre=[]
        cp_matched_repre=[]
        matched_qp_dim=0
        matched_cp_dim=0
        qp_matched_repre.append(tf.reduce_max(qp_cosine_matrix_transpose, axis=2,keep_dims=True))
        qp_matched_repre.append(tf.reduce_mean(qp_cosine_matrix_transpose, axis=2,keep_dims=True))
        cp_matched_repre.append(tf.reduce_max(cp_cosine_matrix_transpose, axis=2,keep_dims=True))
        cp_matched_repre.append(tf.reduce_mean(cp_cosine_matrix_transpose, axis=2,keep_dims=True))
        matched_cp_dim+=2
        matched_qp_dim+=2
    elif matching_option in [4,5,6]:
        pq_matched_repre=[]
        matched_pq_dim=0
        pq_matched_repre.append(tf.reduce_max(qp_cosine_matrix_transpose, axis=2,keep_dims=True))
        pq_matched_repre.append(tf.reduce_mean(qp_cosine_matrix_transpose, axis=2,keep_dims=True))
        matched_pq_dim+=2
        
        if matching_option in [4,5]:
            qp_matched_repre=[]
            matched_qp_dim=0
            qp_matched_repre.append(tf.reduce_max(qp_cosine_matrix, axis=2,keep_dims=True))
            qp_matched_repre.append(tf.reduce_mean(qp_cosine_matrix, axis=2,keep_dims=True))
            matched_qp_dim+=2
        
        if matching_option in [4,6]:
            pc_matched_repre=[]
            matched_pc_dim=0
            pc_matched_repre.append(tf.reduce_max(cp_cosine_matrix_transpose, axis=2,keep_dims=True))
            pc_matched_repre.append(tf.reduce_mean(cp_cosine_matrix_transpose, axis=2,keep_dims=True))
            matched_pc_dim+=2


    # print('here')
    if MP_dim>0:
        if with_max_attentive_match:
            # max_att word level
            def add_max_attentive(matched_representations,in_matching_repres,in_base_repres,cosine_matrix, name):
                max_att = cal_max_question_representation(in_matching_repres, cosine_matrix)# [batch_size, passage_len, dim]
                max_att_decomp_params = tf.get_variable(name, shape=[MP_dim, input_dim], dtype=tf.float32)
                max_attentive_rep = cal_attentive_matching(in_base_repres, max_att, max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
                matched_representations.append(max_attentive_rep)
            if matching_option==0:
                add_max_attentive(matched_passage_representations,in_question_repres,in_passage_repres,qp_cosine_matrix,"qp_word_max_att_decomp_params")
                matched_passage_dim+=MP_dim

                add_max_attentive(matched_passage_representations,in_choice_repres,in_passage_repres,cp_cosine_matrix,"cp_word_max_att_decomp_params")
                matched_passage_dim+=MP_dim
            elif matching_option in [1,2]:
                add_max_attentive(qp_matched_repre,in_question_repres,in_passage_repres,qp_cosine_matrix,"qp_word_max_att_decomp_params")
                matched_qp_dim+=MP_dim

                add_max_attentive(cp_matched_repre,in_choice_repres,in_passage_repres,cp_cosine_matrix,"cp_word_max_att_decomp_params")
                matched_cp_dim+=MP_dim
            elif matching_option==3:
                add_max_attentive(qp_matched_repre,in_passage_repres,in_question_repres,qp_cosine_matrix_transpose,"pq_word_max_att_decomp_params")
                matched_qp_dim+=MP_dim

                add_max_attentive(cp_matched_repre,in_passage_repres,in_choice_repres,cp_cosine_matrix_transpose,"pc_word_max_att_decomp_params")
                matched_cp_dim+=MP_dim
            elif matching_option in [4,5,6]:
                add_max_attentive(pq_matched_repre,in_passage_repres,in_question_repres,qp_cosine_matrix_transpose,"pq_word_max_att_decomp_params")
                matched_pq_dim+=MP_dim
                if matching_option in [4,5]:
                    add_max_attentive(qp_matched_repre,in_question_repres,in_passage_repres,qp_cosine_matrix,"qp_word_max_att_decomp_params")
                    matched_qp_dim+=MP_dim
                if matching_option in [4,6]:
                    add_max_attentive(pc_matched_repre,in_passage_repres,in_choice_repres,cp_cosine_matrix_transpose,"pc_word_max_att_decomp_params")
                    matched_pc_dim+=MP_dim


            # add_max_attentive(matched_question_representations,in_passage_repres,in_question_repres,qp_cosine_matrix_transpose,"pa_word_max_att_decomp_params")
            # matched_question_dim+=MP_dim


    # print('here')
    with tf.variable_scope('context_MP_matching'):
        for i in range(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    if is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])

                    # question representation
                    (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                        sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                    in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

                    # passage representation
                    tf.get_variable_scope().reuse_variables()
                    (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                        sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                    in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)

                    # choice representation
                    tf.get_variable_scope().reuse_variables()
                    (choice_context_representation_fw, choice_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                        context_lstm_cell_fw, context_lstm_cell_bw, in_choice_repres, dtype=tf.float32, 
                                        sequence_length=choice_lengths) # [batch_size, choice_len, context_lstm_dim]
                    in_choice_repres = tf.concat([choice_context_representation_fw, choice_context_representation_bw], 2)

                if with_no_match:
                    print('No match!!!!')
                    if match_to_passage:
                        matched_passage_representations.append(passage_context_representation_fw)
                        matched_passage_representations.append(passage_context_representation_bw)
                    if match_to_question:
                        matched_question_representations.append(question_context_representation_fw)
                        matched_question_representations.append(question_context_representation_bw)
                    if match_to_choice:
                        matched_choice_representations.append(choice_context_representation_fw)
                        matched_choice_representations.append(choice_context_representation_bw)

                    # question_aware_representatins.append(question_context_representation_fw)
                    # print(question_context_representation_fw.shape)
                    # print(question_context_representation_bw.shape)
                    # question_aware_representatins.append(question_context_representation_bw)
                    # passage_aware_representatins.append(passage_context_representation_fw)
                    # passage_aware_representatins.append(passage_context_representation_bw)
                else:
                    # First step matching
                    # Multi-perspective matching
                    if matching_option in [0,1,2]:
                        with tf.variable_scope('qp_MP_matching'):
                            (qp_matching_vectors_fw, qp_matching_vectors_bw, qp_matching_fw_dim, qp_matching_bw_dim) = \
                                    match_passage_with_question_direct(passage_context_representation_fw, 
                                        passage_context_representation_bw, mask,
                                        question_context_representation_fw, question_context_representation_bw,question_mask,
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            if matching_option in [0,1]:
                                qp_matching_vectors_fw=tf.concat(qp_matching_vectors_fw,2)
                                qp_matching_vectors_bw=tf.concat(qp_matching_vectors_bw,2)

                        # Multi-perspective matching
                        with tf.variable_scope('cp_MP_matching'):
                            (cp_matching_vectors_fw, cp_matching_vectors_bw, cp_matching_fw_dim, cp_matching_bw_dim) = \
                                    match_passage_with_question_direct(passage_context_representation_fw,
                                        passage_context_representation_bw, mask,
                                        choice_context_representation_fw, choice_context_representation_bw,choice_mask,
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            if matching_option in [0,1]:
                                cp_matching_vectors_fw=tf.concat(cp_matching_vectors_fw,2)
                                cp_matching_vectors_bw=tf.concat(cp_matching_vectors_bw,2)
                        if matching_option in [0,1]:
                            matching_tensors=[passage_context_representation_fw,passage_context_representation_bw,
                             question_context_representation_fw, question_context_representation_fw, 
                             choice_context_representation_fw, choice_context_representation_bw, 
                             qp_matching_vectors_fw,qp_matching_vectors_bw,cp_matching_vectors_fw,cp_matching_vectors_bw]
                        else:
                            matching_tensors=[passage_context_representation_fw,passage_context_representation_bw, 
                                question_context_representation_fw,question_context_representation_bw, 
                                choice_context_representation_fw,choice_context_representation_bw]
                    elif matching_option==3:
                        with tf.variable_scope('pq_MP_matching'):
                            (qp_matching_vectors_fw, qp_matching_vectors_bw, qp_matching_fw_dim, qp_matching_bw_dim) = \
                                    match_passage_with_question_direct(question_context_representation_fw, 
                                        question_context_representation_bw,question_mask,
                                        passage_context_representation_fw, passage_context_representation_bw, mask,                                        
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            qp_matching_vectors_fw=tf.concat(qp_matching_vectors_fw,2)
                            qp_matching_vectors_bw=tf.concat(qp_matching_vectors_bw,2)

                        # Multi-perspective matching
                        with tf.variable_scope('pc_MP_matching'):
                            (cp_matching_vectors_fw, cp_matching_vectors_bw, cp_matching_fw_dim, cp_matching_bw_dim) = \
                                    match_passage_with_question_direct(choice_context_representation_fw, 
                                        choice_context_representation_bw,choice_mask,
                                        passage_context_representation_fw, passage_context_representation_bw, mask,                                        
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            cp_matching_vectors_fw=tf.concat(cp_matching_vectors_fw,2)
                            cp_matching_vectors_bw=tf.concat(cp_matching_vectors_bw,2)
                            matching_tensors=[passage_context_representation_fw,passage_context_representation_bw,
                             question_context_representation_fw, question_context_representation_bw, 
                             choice_context_representation_fw, choice_context_representation_bw, 
                             qp_matching_vectors_fw,qp_matching_vectors_bw,cp_matching_vectors_fw,cp_matching_vectors_bw]
                    elif matching_option in [4,5,6]:
                        matching_tensors=[passage_context_representation_fw,passage_context_representation_bw, 
                                    question_context_representation_fw,question_context_representation_bw, 
                                    choice_context_representation_fw,choice_context_representation_bw]
                        if matching_option in [4,5]:
                            with tf.variable_scope('qp_MP_matching'):
                                (qp_matching_vectors_fw, qp_matching_vectors_bw, qp_matching_fw_dim, qp_matching_bw_dim) = \
                                        match_passage_with_question_direct(passage_context_representation_fw, 
                                            passage_context_representation_bw, mask,
                                            question_context_representation_fw, question_context_representation_bw,question_mask,
                                            MP_dim, context_lstm_dim, scope=None,
                                            with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                            with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                            with_direction=True)
                                if matching_option==5:
                                    qp_matching_vectors_fw=tf.concat(qp_matching_vectors_fw,2)
                                    qp_matching_vectors_bw=tf.concat(qp_matching_vectors_bw,2)
                                    matching_tensors.append(qp_matching_vectors_fw)                        
                                    matching_tensors.append(qp_matching_vectors_bw)                        
                        with tf.variable_scope('pq_MP_matching'):
                            (pq_matching_vectors_fw, pq_matching_vectors_bw, pq_matching_fw_dim, pq_matching_bw_dim) = \
                                    match_passage_with_question_direct(question_context_representation_fw, 
                                        question_context_representation_bw,question_mask,
                                        passage_context_representation_fw, passage_context_representation_bw, mask,                                        
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            if matching_option==5:
                                pq_matching_vectors_fw=tf.concat(pq_matching_vectors_fw,2)
                                pq_matching_vectors_bw=tf.concat(pq_matching_vectors_bw,2)
                                matching_tensors.append(pq_matching_vectors_fw) 
                                matching_tensors.append(pq_matching_vectors_bw) 
                        with tf.variable_scope('pc_MP_matching'):
                            (pc_matching_vectors_fw, pc_matching_vectors_bw, pc_matching_fw_dim, pc_matching_bw_dim) = \
                                    match_passage_with_question_direct(choice_context_representation_fw, 
                                        choice_context_representation_bw,choice_mask,
                                        passage_context_representation_fw, passage_context_representation_bw, mask,                                        
                                        MP_dim, context_lstm_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                        with_direction=True)
                            if matching_option==5:
                                pc_matching_vectors_fw=tf.concat(pc_matching_vectors_fw,2)
                                pc_matching_vectors_bw=tf.concat(pc_matching_vectors_bw,2)
                                matching_tensors.append(pc_matching_vectors_fw)
                                matching_tensors.append(pc_matching_vectors_bw)
                    # Post matching
                    if matching_option in [0,1]:
                        # Multi-perspective matching
                        with tf.variable_scope('qc_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(qp_matching_vectors_fw, 
                                        qp_matching_vectors_bw, mask,
                                        cp_matching_vectors_fw, cp_matching_vectors_bw, mask,
                                        MP_dim, cp_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            if matching_option==0:
                                matched_passage_representations.extend(matching_vectors)
                                matched_passage_dim += matching_dim
                            elif matching_option==1:
                                qp_matched_repre.extend(matching_vectors)
                                matched_qp_dim += matching_dim
                        # Multi-perspective matching
                        with tf.variable_scope('cq_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(cp_matching_vectors_fw, 
                                        cp_matching_vectors_bw, mask,
                                        qp_matching_vectors_fw, qp_matching_vectors_bw, mask,
                                        MP_dim, cp_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            if matching_option==0:
                                matched_passage_representations.extend(matching_vectors)
                                matched_passage_dim += matching_dim
                            elif matching_option==1:
                                cp_matched_repre.extend(matching_vectors)
                                matched_cp_dim += matching_dim
                    elif matching_option==2:
                        qp_matched_repre.extend(qp_matching_vectors_fw+qp_matching_vectors_bw)
                        matched_qp_dim += qp_matching_fw_dim+qp_matching_bw_dim
                        cp_matched_repre.extend(cp_matching_vectors_fw+cp_matching_vectors_bw)
                        matched_cp_dim += cp_matching_fw_dim+cp_matching_bw_dim
                    elif matching_option==3:
                        # Multi-perspective matching
                        with tf.variable_scope('qc_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(qp_matching_vectors_fw, 
                                        qp_matching_vectors_bw, question_mask,
                                        cp_matching_vectors_fw, cp_matching_vectors_bw, choice_mask,
                                        MP_dim, cp_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            qp_matched_repre.extend(matching_vectors)
                            matched_qp_dim += matching_dim
                        # Multi-perspective matching
                        with tf.variable_scope('cq_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(cp_matching_vectors_fw, 
                                        cp_matching_vectors_bw, choice_mask,
                                        qp_matching_vectors_fw, qp_matching_vectors_bw, question_mask,
                                        MP_dim, cp_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            cp_matched_repre.extend(matching_vectors)
                            matched_cp_dim += matching_dim
                    elif matching_option in [4,6]:
                        if matching_option==4:
                            qp_matched_repre.extend(qp_matching_vectors_fw+qp_matching_vectors_bw)
                            matched_qp_dim+= qp_matching_fw_dim+qp_matching_bw_dim
                        pq_matched_repre.extend(pq_matching_vectors_fw+pq_matching_vectors_bw)
                        matched_pq_dim+= pq_matching_fw_dim+pq_matching_bw_dim
                        pc_matched_repre.extend(pc_matching_vectors_fw+pc_matching_vectors_bw)
                        matched_pc_dim+= pc_matching_fw_dim+pc_matching_bw_dim
                    elif matching_option==5:
                        # Multi-perspective matching
                        with tf.variable_scope('cp_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(qp_matching_vectors_fw, 
                                        qp_matching_vectors_bw, mask,
                                        pc_matching_vectors_fw, pc_matching_vectors_bw, choice_mask, 
                                        MP_dim, pc_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            qp_matched_repre.extend(matching_vectors)
                            matched_qp_dim += matching_dim
                        # Multi-perspective matching
                        with tf.variable_scope('cq_post_MP_matching'):
                            (matching_vectors, matching_dim) = match_passage_with_question_direct(pq_matching_vectors_fw, 
                                        pq_matching_vectors_bw, question_mask,
                                        pc_matching_vectors_fw, pc_matching_vectors_bw, choice_mask,
                                        MP_dim, pc_matching_fw_dim, scope=None,
                                        with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                        with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                            pq_matched_repre.extend(matching_vectors)
                            matched_pq_dim += matching_dim                        
    # print('here333')
    if matching_option==0:
        matching_tensors.extend(matched_passage_representations)
        matched_passage_representations = tf.concat(matched_passage_representations, 2) # [batch_size, passage_len, dim]
        matching_tensors.append(matched_passage_representations)

        if is_training:
            matched_passage_representations = tf.nn.dropout(matched_passage_representations, (1 - dropout_rate))
        else:
            matched_passage_representations = tf.multiply(matched_passage_representations, (1 - dropout_rate))
            
        # ======Highway layer======
        if with_match_highway:
            with tf.variable_scope("matching_highway"):
                matched_passage_representations = multi_highway_layer(matched_passage_representations, matched_passage_dim, highway_layer_num)
            
        #========Aggregation Layer======
        aggregation_representation = []
        aggregation_dim = 0
        aggregation_input = matched_passage_representations
        with tf.variable_scope('aggregation_layer'):
            for i in range(aggregation_layer_num):
                with tf.variable_scope('layer-{}'.format(i)):
                    aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                    if is_training:
                        aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                    aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])

                    cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                            aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_input, 
                            dtype=tf.float32, sequence_length=passage_lengths)

                    fw_rep = cur_aggregation_representation[0][:,-1,:]
                    bw_rep = cur_aggregation_representation[1][:,0,:]
                    aggregation_representation.append(fw_rep)
                    aggregation_representation.append(bw_rep)
                    aggregation_dim += 2* aggregation_lstm_dim
                    aggregation_input = tf.concat(cur_aggregation_representation, 2)# [batch_size, passage_len, 2*aggregation_lstm_dim]
            
        #
        aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

        # ======Highway layer======
        if with_aggregation_highway:
            with tf.variable_scope("aggregation_highway"):
                agg_shape = tf.shape(aggregation_representation)
                batch_size = agg_shape[0]
                aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
                aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
                aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
        if debug:
            return (aggregation_representation,aggregation_dim,matching_tensors)
        return (aggregation_representation, aggregation_dim)
    else:
        if matching_option in [1,2,3]:
            matching_tensors.extend(qp_matched_repre)
            matching_tensors.extend(cp_matched_repre)
            left_repres=tf.concat(qp_matched_repre,2)
            right_repres = tf.concat(cp_matched_repre,2)
            matching_tensors.append(left_repres)
            matching_tensors.append(right_repres)
            left_dim=matched_qp_dim
            right_dim=matched_cp_dim
            print('left dim=',left_dim,'right dim=',right_dim)

            if is_training:
                left_repres = tf.nn.dropout(left_repres, (1 - dropout_rate))
                right_repres = tf.nn.dropout(right_repres, (1 - dropout_rate))
            else:
                left_repres = tf.multiply(left_repres, (1 - dropout_rate))
                right_repres = tf.multiply(right_repres, (1 - dropout_rate))
                
            # ======Highway layer======
            if with_match_highway:
                with tf.variable_scope("left_matching_highway"):
                    left_repres = multi_highway_layer(left_repres, left_dim,highway_layer_num)
                with tf.variable_scope("right_matching_highway"):
                    right_repres = multi_highway_layer(right_repres, right_dim,highway_layer_num)

            aggregation_inputs=[left_repres,right_repres]
            aggregation_dims = [left_dim, right_dim]
            if matching_option in [1,2]:
                aggregation_lengths=[passage_lengths,passage_lengths]
            else:
                aggregation_lengths=[question_lengths,choice_lengths]
        elif matching_option in [4,6]:
            # print('aggregate')
            if matching_option==4:
                matching_tensors.extend(qp_matched_repre)
                repres_1=tf.concat(qp_matched_repre,2)
                matching_tensors.append(repres_1)
                rep1_dim=matched_qp_dim
                if is_training:
                    repres_1 = tf.nn.dropout(repres_1, (1 - dropout_rate))
                else:
                    repres_1 = tf.multiply(repres_1, (1 - dropout_rate))
            else:
                rep1_dim=0
            
            matching_tensors.extend(pq_matched_repre)
            matching_tensors.extend(pc_matched_repre)
            repres_2=tf.concat(pq_matched_repre,2)
            repres_3=tf.concat(pc_matched_repre,2)
            matching_tensors.append(repres_2)
            matching_tensors.append(repres_3)
            rep2_dim=matched_pq_dim
            rep3_dim=matched_pc_dim
            print('dims:',rep1_dim,rep2_dim,rep3_dim)
            if is_training:
                repres_2 = tf.nn.dropout(repres_2, (1 - dropout_rate))
                repres_3 = tf.nn.dropout(repres_3, (1 - dropout_rate))
            else:
                repres_2 = tf.multiply(repres_2, (1 - dropout_rate))
                repres_3 = tf.multiply(repres_3, (1 - dropout_rate))
                
            # ======Highway layer======
            if with_match_highway:
                if matching_option==4:
                    with tf.variable_scope("matching_highway_1"):
                        repres_1 = multi_highway_layer(repres_1, rep1_dim, highway_layer_num)
                with tf.variable_scope("matching_highway_2"):
                    repres_2 = multi_highway_layer(repres_2, rep2_dim, highway_layer_num)
                with tf.variable_scope("matching_highway_3"):
                    repres_3 = multi_highway_layer(repres_3, rep3_dim, highway_layer_num)    
            if matching_option==4:
                aggregation_inputs=[repres_1,repres_2,repres_3]
                aggregation_dims=[rep1_dim,rep2_dim,rep3_dim]
                aggregation_lengths=[passage_lengths,question_lengths,choice_lengths]
            else:
                aggregation_inputs=[repres_2,repres_3]
                aggregation_dims=[rep2_dim,rep3_dim]
                aggregation_lengths=[question_lengths,choice_lengths]

        elif matching_option==5:
            matching_tensors.extend(qp_matched_repre)
            matching_tensors.extend(pq_matched_repre)
            repres_1=tf.concat(qp_matched_repre,2)
            repres_2=tf.concat(pq_matched_repre,2)
            rep1_dim=matched_qp_dim
            rep2_dim=matched_pq_dim
            matching_tensors.append(repres_1)
            matching_tensors.append(repres_2)

            if is_training:
                repres_1 = tf.nn.dropout(repres_1, (1 - dropout_rate))
                repres_2 = tf.nn.dropout(repres_2, (1 - dropout_rate))
            else:
                repres_1 = tf.multiply(repres_1, (1 - dropout_rate))
                repres_2 = tf.multiply(repres_2, (1 - dropout_rate))
                
            # ======Highway layer======
            if with_match_highway:
                with tf.variable_scope("matching_highway_1"):
                    repres_1 = multi_highway_layer(repres_1, rep1_dim, highway_layer_num)
                with tf.variable_scope("matching_highway_2"):
                    repres_2 = multi_highway_layer(repres_2, rep2_dim, highway_layer_num)
            aggregation_inputs=[repres_1,repres_2]
            aggregation_dims=[rep1_dim,rep2_dim]
            aggregation_lengths=[passage_lengths,question_lengths]            
            
        #========Aggregation Layer======
        aggregation_representation = []
        aggregation_dim = 0
        print('aggregation dims:',aggregation_dims)
        
        '''
        if with_mean_aggregation:
            aggregation_representation.append(tf.reduce_mean(left_repres, axis=1))
            aggregation_dim += left_dim
            aggregation_representation.append(tf.reduce_mean(right_repres, axis=1))
            aggregation_dim += right_dim
        #'''
        with tf.variable_scope('aggregation_layer'):
            for i in range(aggregation_layer_num): # support multiple aggregation layer
                for rep_id in range(len(aggregation_inputs)):
                    with tf.variable_scope('layer-{}-{}'.format(rep_id, i)):
                        aggregation_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                        aggregation_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(aggregation_lstm_dim)
                        if is_training:
                            aggregation_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                            aggregation_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(aggregation_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                        aggregation_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_fw])
                        aggregation_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([aggregation_lstm_cell_bw])
                        cur_aggregation_representation, _ = my_rnn.bidirectional_dynamic_rnn(
                                aggregation_lstm_cell_fw, aggregation_lstm_cell_bw, aggregation_inputs[rep_id], 
                                dtype=tf.float32, sequence_length=aggregation_lengths[rep_id])                      

                        fw_rep = cur_aggregation_representation[0][:,-1,:]
                        bw_rep = cur_aggregation_representation[1][:,0,:]
                        aggregation_representation.append(fw_rep)
                        aggregation_representation.append(bw_rep)
                        aggregation_dim += 2* aggregation_lstm_dim
        #
        aggregation_representation = tf.concat(aggregation_representation, 1) # [batch_size, aggregation_dim]

        # ======Highway layer======
        if with_aggregation_highway:
            with tf.variable_scope("aggregation_highway"):
                agg_shape = tf.shape(aggregation_representation)
                batch_size = agg_shape[0]
                aggregation_representation = tf.reshape(aggregation_representation, [1, batch_size, aggregation_dim])
                aggregation_representation = multi_highway_layer(aggregation_representation, aggregation_dim, highway_layer_num)
                aggregation_representation = tf.reshape(aggregation_representation, [batch_size, aggregation_dim])
        
        if debug:
            return (aggregation_representation,aggregation_dim,matching_tensors)
        return (aggregation_representation, aggregation_dim)


# def gated_trilateral_match(in_question_repres, in_passage_repres, in_choice_repres,
#                         question_lengths, passage_lengths, choice_lengths, 
#                         question_mask, mask, choice_mask, 
#                         concat_idx_mat, split_idx_mat_q, split_idx_mat_c, 
#                         MP_dim, input_dim, context_layer_num, context_lstm_dim,is_training,dropout_rate,
#                         with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
#                         with_aggregation_highway, with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
#                         with_max_attentive_match=True, with_no_match=False, 
#                         concat_context=False, tied_aggre=True, rl_matches=[0,1,2], debug=False):

#     '''
#     rl_matches options:
#     0: a=p->(q+c), split a->[a1,a2]
#     1: p->c, [q,c]
#     2: a1=p->c, a2=p->q, [a1->a2, a2->a1]
#     '''
#     matching_tensors=[]
#     print('gated trilateral match')
#     qp_cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
#     qp_cosine_matrix = mask_relevancy_matrix(qp_cosine_matrix, question_mask, mask)
#     qp_cosine_matrix_transpose = tf.transpose(qp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

#     cp_cosine_matrix = cal_relevancy_matrix(in_choice_repres, in_passage_repres) # [batch_size, passage_len, question_len]
#     cp_cosine_matrix = mask_relevancy_matrix(cp_cosine_matrix, choice_mask, mask)
#     cp_cosine_matrix_transpose = tf.transpose(cp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

#     word_level_max_pooling_pq = tf.reduce_max(qp_cosine_matrix_transpose, axis=2,keep_dims=True)
#     word_level_avg_pooling_pq = tf.reduce_mean(qp_cosine_matrix_transpose, axis=2,keep_dims=True)
#     word_level_max_pooling_pc = tf.reduce_max(cp_cosine_matrix_transpose, axis=2,keep_dims=True)
#     word_level_avg_pooling_pc = tf.reduce_mean(cp_cosine_matrix_transpose, axis=2,keep_dims=True)

#     if MP_dim>0 and with_max_attentive_match:
#         def max_attentive(in_matching_repres,in_base_repres,cosine_matrix, name):
#             max_att = cal_max_question_representation(in_matching_repres, cosine_matrix)# [batch_size, passage_len, dim]
#             max_att_decomp_params = tf.get_variable(name, shape=[MP_dim, input_dim], dtype=tf.float32)
#             max_attentive_rep = cal_attentive_matching(in_base_repres, max_att, max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
#             return max_attentive_rep
#         word_level_max_attentive_pq=max_attentive(in_passage_repres,in_question_repres,qp_cosine_matrix_transpose,"pq_word_max_att_decomp_params")
#         word_level_max_attentive_pc=max_attentive(in_passage_repres,in_choice_repres,cp_cosine_matrix_transpose,"pc_word_max_att_decomp_params")

#     if 0 in rl_matches:
#         if MP_dim>0 and with_max_attentive_match:
#             question_concat_basic_embedding=tf.concat([word_level_max_pooling_pq,word_level_avg_pooling_pq,word_level_max_attentive_pq],2)
#             choice_concat_basic_embedding=tf.concat([word_level_max_pooling_pc,word_level_avg_pooling_pc,word_level_max_attentive_pc],2)
#             qc_basic_dim=4+2*MP_dim
#         else:
#             question_concat_basic_embedding=tf.concat([word_level_max_pooling_pq,word_level_avg_pooling_pq],2)
#             choice_concat_basic_embedding=tf.concat([word_level_max_pooling_pc,word_level_avg_pooling_pc],2)
#             qc_basic_dim=4
#         qc_basic_embedding=my_rnn.concatenate_sents(question_concat_basic_embedding,choice_concat_basic_embedding, concat_idx_mat)



#     all_match_templates=[]
#     for matchid in rl_matches:
#         if matchid in [1,2]:
#             matcher=Matcher(matchid, question_lengths, choice_lengths)
#             matcher.add_question_repre(word_level_max_pooling_pq,1)
#             matcher.add_question_repre(word_level_avg_pooling_pq,1)
#             matcher.add_choice_repre(word_level_max_pooling_pc,1)
#             matcher.add_choice_repre(word_level_avg_pooling_pc,1)

#             if MP_dim>0 and with_max_attentive_match:
#                 matcher.add_question_repre(word_level_max_attentive_pq, MP_dim)
#                 matcher.add_choice_repre(word_level_max_attentive_pc,MP_dim)

#             all_match_templates.append(matcher)
#         elif matchid==0:
#             matcher=Matcher(matchid, question_lengths, choice_lengths)
#             matcher.add_question_repre(qc_basic_embedding,qc_basic_dim)
#             all_match_templates.append(matcher)


#     # print('here')
#     with tf.variable_scope('context_MP_matching'):
#         for i in range(context_layer_num): # support multiple context layer
#             with tf.variable_scope('layer-{}'.format(i)):
#                 with tf.variable_scope('context_represent'):
#                     # parameters
#                     context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
#                     context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
#                     if is_training:
#                         context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
#                         context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
#                     context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
#                     context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])




#                     if concat_context:
#                         in_qc_repres=my_rnn.concatenate_sents(in_question_repres,in_choice_repres,concat_idx_mat)
#                         # question representation
#                         (qc_context_representation_fw, qc_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                             context_lstm_cell_fw, context_lstm_cell_bw, in_qc_repres, dtype=tf.float32, 
#                                             sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
#                         # in_qc_repres = tf.concat([qc_context_representation_fw, qc_context_representation_bw], 2)

#                         # gate_input=my_rnn.extract_question_repre(qc_context_representation_fw,qc_context_representation_bw,question_lengths)
#                         question_context_representation_fw, choice_context_representation_fw = \
#                             my_rnn.split_sents(qc_context_representation_fw,split_idx_mat_q, split_idx_mat_c)
#                         question_context_representation_bw, choice_context_representation_bw = \
#                             my_rnn.split_sents(qc_context_representation_bw, split_idx_mat_q, split_idx_mat_c)



#                         # passage representation
#                         tf.get_variable_scope().reuse_variables()
#                         (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                             context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
#                                             sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
#                         in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)
#                     else:
#                         # question representation
#                         (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                             context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
#                                             sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
#                         in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

#                         gate_input=[question_context_representation_fw[:,-1,:],question_context_representation_bw[:,0,:]]
#                         # passage representation
#                         tf.get_variable_scope().reuse_variables()
#                         (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                             context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
#                                             sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
#                         in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)

#                         # choice representation
#                         tf.get_variable_scope().reuse_variables()
#                         (choice_context_representation_fw, choice_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
#                                             context_lstm_cell_fw, context_lstm_cell_bw, in_choice_repres, dtype=tf.float32, 
#                                             sequence_length=choice_lengths) # [batch_size, choice_len, context_lstm_dim]
#                         in_choice_repres = tf.concat([choice_context_representation_fw, choice_context_representation_bw], 2)

#                         if 0 in rl_matches:
#                             qc_context_representation_fw = my_rnn.concatenate_sents(question_context_representation_fw, 
#                                 choice_context_representation_fw, concat_idx_mat)
#                             qc_context_representation_bw = my_rnn.concatenate_sents(question_context_representation_bw, 
#                                 choice_context_representation_bw, concat_idx_mat)
#                             # in_qc_repres=my_rnn.concatenate_sents(in_question_repres,in_choice_repres,concat_idx_mat)
#                     if 0 in rl_matches:
#                         qc_lengths=question_lengths+choice_lengths
#                         qc_shape=tf.shape(qc_context_representation_fw)
#                         qc_len=qc_shape[1]
#                         qc_mask = tf.sequence_mask(qc_lengths, qc_len, dtype=tf.float32)

#                 with tf.variable_scope('rl_decision_gate'):
#                     gate_input=tf.concat([question_context_representation_fw[:,-1,:],question_context_representation_bw[:,0,:]],1)
#                     w_gate=tf.get_variable('w_gate',[2*context_lstm_dim,len(rl_matches)],dtype=tf.float32)
#                     b_gate=tf.get_variable('b_gate',[len(rl_matches)],dtype=tf.float32)
#                     gate_logits=tf.matmul(gate_input,w_gate)+b_gate

#                     gate_prob=tf.nn.softmax(gate_logits)

#                     gate_log_prob=tf.nn.log_softmax(gate_logits)

#                 if 0 in rl_matches:
#                     with tf.variable_scope('p_qc_matching'):
#                         (p_qc_matching_vectors,p_qc_matching_dim) = match_passage_with_question(passage_context_representation_fw, 
#                                 passage_context_representation_bw, mask,
#                                 qc_context_representation_fw, qc_context_representation_bw,qc_mask,
#                                 MP_dim, context_lstm_dim, scope=None,
#                                 with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
#                                 with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
#                 if 1 in rl_matches or 2 in rl_matches:
#                     with tf.variable_scope('p_c_matching'):
#                         (p_c_matching_vectors_fw, p_c_matching_vectors_bw, p_c_matching_dim_fw, p_c_matching_dim_bw) = \
#                                 match_passage_with_question_direct(passage_context_representation_fw, 
#                                 passage_context_representation_bw, mask,
#                                 choice_context_representation_fw, choice_context_representation_bw,choice_mask,
#                                 MP_dim, context_lstm_dim, scope=None,
#                                 with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
#                                 with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
#                                 with_direction=True)
#                 if 2 in rl_matches:
#                     with tf.variable_scope('p_q_matching'):
#                         (p_q_matching_vectors_fw, p_q_matching_vectors_bw, p_q_matching_dim_fw, p_q_matching_dim_bw) = \
#                                 match_passage_with_question_direct(passage_context_representation_fw, 
#                                 passage_context_representation_bw, mask,
#                                 question_context_representation_fw, question_context_representation_bw,question_mask,
#                                 MP_dim, context_lstm_dim, scope=None,
#                                 with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
#                                 with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
#                                 with_direction=True)
#                     # Multi-perspective matching
#                     p_c_matching_vectors_fw_concat=tf.concat(p_c_matching_vectors_fw,2)
#                     p_c_matching_vectors_bw_concat=tf.concat(p_c_matching_vectors_bw,2)
#                     p_q_matching_vectors_fw_concat=tf.concat(p_q_matching_vectors_fw,2)
#                     p_q_matching_vectors_bw_concat=tf.concat(p_q_matching_vectors_bw,2)

#                     with tf.variable_scope('cq_post_MP_matching'):
#                         (c_q_postmatching_vectors, c_q_postmatching_dim) = match_passage_with_question_direct(
#                                     p_c_matching_vectors_fw_concat, p_c_matching_vectors_bw_concat, choice_mask,
#                                     p_q_matching_vectors_fw_concat, p_q_matching_vectors_bw_concat, question_mask,
#                                     MP_dim, p_q_matching_dim_fw, scope=None,
#                                     with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
#                                     with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
#                     with tf.variable_scope('qc_post_MP_matching'):
#                         (q_c_postmatching_vectors, q_c_postmatching_dim) = match_passage_with_question_direct(
#                                     p_q_matching_vectors_fw_concat, p_q_matching_vectors_bw_concat, question_mask,
#                                     p_c_matching_vectors_fw_concat, p_c_matching_vectors_bw_concat, choice_mask,
#                                     MP_dim, p_q_matching_dim_fw, scope=None,
#                                     with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
#                                     with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
#                 aggre_defined=False
                
#                 for (mat_id,rl_match_opt) in enumerate(rl_matches):
#                     current_matcher=all_match_templates[mat_id]
#                     if rl_match_opt==0:
#                         current_matcher.add_question_repre(p_qc_matching_vectors,p_qc_matching_dim,extend=True)
#                     if rl_match_opt==1:
#                         current_matcher.add_question_repre(question_context_representation_fw, context_lstm_dim)
#                         current_matcher.add_question_repre(question_context_representation_bw, context_lstm_dim)
#                         if 2 in rl_matches:
#                             current_matcher.add_choice_repre(p_c_matching_vectors_fw_concat, p_c_matching_dim_fw)
#                             current_matcher.add_choice_repre(p_c_matching_vectors_bw_concat, p_c_matching_dim_bw)
#                         else:
#                             current_matcher.add_choice_repre(p_c_matching_vectors_fw, p_c_matching_dim_fw,extend=True)
#                             current_matcher.add_choice_repre(p_c_matching_vectors_bw, p_c_matching_dim_bw,extend=True)
#                     if rl_match_opt==2:
#                         current_matcher.add_question_repre(c_q_postmatching_vectors,c_q_postmatching_dim, extend=True)
#                         current_matcher.add_choice_repre(q_c_postmatching_vectors,q_c_postmatching_dim, extend=True)
    
#     # TODO: add tied LSTM weights
#     added_agg_highway=False

#     for mid,matcher in enumerate(all_match_templates):
#         matching_tensors.extend(matcher.question_repre)
#         matching_tensors.extend(matcher.choice_repre)
#         matcher.concat(is_training,dropout_rate)
#         if matcher.question_repre_dim>0:
#             matching_tensors.append(matcher.question_repre)
#         if matcher.choice_repre_dim>0:
#             matching_tensors.append(matcher.choice_repre)

#         if with_match_highway:
#             matcher.add_highway_layer(highway_layer_num, 'highway_{}'.format(mid) ,reuse=False)
#         agg_dim=matcher.aggregate('aggregate_{}'.format(mid), aggregation_layer_num, aggregation_lstm_dim, dropout_rate, reuse=False)
#         print('aggregation dim=',agg_dim)
#         if with_aggregation_highway:
#             if not added_agg_highway:
#                 matcher.add_aggregation_highway(highway_layer_num, 'aggregation_highway', reuse=False)
#                 added_agg_highway=True
#             else:
#                 matcher.add_aggregation_highway(highway_layer_num, 'aggregation_highway', reuse=True)

#     if verbose:
#         return all_match_templates, agg_dim, gate_prob, gate_log_prob, matching_tensors
#     else:
#         return all_match_templates, agg_dim, gate_prob, gate_log_prob