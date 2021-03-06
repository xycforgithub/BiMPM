import tensorflow as tf
from tensorflow.python.ops import rnn
import my_rnn
from matcher import Matcher
from match_utils import *
from my_rnn import SwitchableDropoutWrapper
from memory import Memory

num_option=4
def maybe_tile(in_tensor, efficient):
    if efficient:
        rank=len(in_tensor.get_shape())
        tilenum=[1 for i in range(rank)]
        tilenum[0]=num_option
        return tf.tile(in_tensor,tilenum)
    else:
        return in_tensor

def gated_trilateral_match(in_question_repres, in_passage_repres, in_choice_repres,
                        question_lengths, passage_lengths, choice_lengths, 
                        question_mask, mask, choice_mask, 
                        concat_idx_mat, split_idx_mat_q, split_idx_mat_c, 
                        MP_dim, input_dim, context_layer_num, context_lstm_dim,is_training,dropout_rate,
                        with_match_highway,aggregation_layer_num, aggregation_lstm_dim,highway_layer_num,
                        with_aggregation_highway, with_full_match=True, with_maxpool_match=True, with_attentive_match=True,
                        with_max_attentive_match=True,
                        concat_context=False, tied_aggre=False, rl_matches=[0,1,2], cond_training=False, efficient = False, 
                        tied_match=False, construct_memory=False, debug=False):

    '''
    rl_matches options:
    0: a=p->(q+c), split a->[a1,a2]
    1: p->c, [q,c]
    2: a1=p->c, a2=p->q, [a1->a2, a2->a1]
    '''
    matching_tensors=[]
    print('gated trilateral match')
    print('concat context=',concat_context)
    print('tied_aggre',tied_aggre)
    qp_cosine_matrix = cal_relevancy_matrix(in_question_repres, in_passage_repres) # [batch_size, passage_len, question_len]
    qp_cosine_matrix = mask_relevancy_matrix(qp_cosine_matrix, question_mask, mask)
    qp_cosine_matrix_transpose = tf.transpose(qp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]
    in_question_shape=tf.shape(in_question_repres)
    question_len=in_question_shape[1]


    tiled_in_passage_repres=maybe_tile(in_passage_repres,efficient)
    tiled_in_question_repres = maybe_tile(in_question_repres, efficient)
    tiled_mask=maybe_tile(mask,efficient)
    tiled_question_mask=maybe_tile(question_mask,efficient)
    tiled_question_lengths=maybe_tile(question_lengths,efficient)
    if 0 in rl_matches:
        qc_lengths=tiled_question_lengths+choice_lengths

    # if efficient:
    #     tiled_in_passage_repres=tf.tile(in_passage_repres,[num_option,1,1])
    #     tiled_mask=tf.tile(tiled_mask,[num_option,1])
    # else:
    #     tiled_in_passage_repres=in_passage_repres
    #     tiled_mask=mask

    cp_cosine_matrix = cal_relevancy_matrix(in_choice_repres, tiled_in_passage_repres) # [batch_size, passage_len, question_len]
    cp_cosine_matrix = mask_relevancy_matrix(cp_cosine_matrix, choice_mask, tiled_mask)
    cp_cosine_matrix_transpose = tf.transpose(cp_cosine_matrix, perm=[0,2,1])# [batch_size, question_len, passage_len]

    word_level_max_pooling_pq = tf.reduce_max(qp_cosine_matrix_transpose, axis=2,keep_dims=True)
    word_level_avg_pooling_pq = tf.reduce_mean(qp_cosine_matrix_transpose, axis=2,keep_dims=True)
    word_level_max_pooling_pq = maybe_tile(word_level_max_pooling_pq, efficient)
    word_level_avg_pooling_pq = maybe_tile(word_level_avg_pooling_pq, efficient)
    untiled_word_level_max_pooling_qp=tf.reduce_max(qp_cosine_matrix, axis=2,keep_dims=True)
    untiled_word_level_avg_pooling_qp=tf.reduce_mean(qp_cosine_matrix, axis=2,keep_dims=True)



    word_level_max_pooling_pc = tf.reduce_max(cp_cosine_matrix_transpose, axis=2,keep_dims=True)
    word_level_avg_pooling_pc = tf.reduce_mean(cp_cosine_matrix_transpose, axis=2,keep_dims=True)

    if MP_dim>0 and with_max_attentive_match:
        def max_attentive(in_matching_repres,in_base_repres,cosine_matrix, name):
            max_att = cal_max_question_representation(in_matching_repres, cosine_matrix)# [batch_size, passage_len, dim]
            max_att_decomp_params = tf.get_variable(name, shape=[MP_dim, input_dim], dtype=tf.float32)
            max_attentive_rep = cal_attentive_matching(in_base_repres, max_att, max_att_decomp_params)# [batch_size, passage_len, decompse_dim]
            return max_attentive_rep
        word_level_max_attentive_pq=max_attentive(in_passage_repres,in_question_repres,qp_cosine_matrix_transpose,"pq_word_max_att_decomp_params")
        untiled_word_level_max_attentive_qp = max_attentive(in_question_repres, in_passage_repres, qp_cosine_matrix,
                                                    "qp_word_max_att_decomp_params")
        word_level_max_attentive_pc=max_attentive(tiled_in_passage_repres,in_choice_repres,cp_cosine_matrix_transpose,"pc_word_max_att_decomp_params")
        word_level_max_attentive_pq=maybe_tile(word_level_max_attentive_pq,efficient)


    if 0 in rl_matches:
        if MP_dim>0 and with_max_attentive_match:
            question_concat_basic_embedding=tf.concat([word_level_max_pooling_pq,word_level_avg_pooling_pq,word_level_max_attentive_pq],2)
            choice_concat_basic_embedding=tf.concat([word_level_max_pooling_pc,word_level_avg_pooling_pc,word_level_max_attentive_pc],2)
            qc_basic_dim=2+MP_dim
        else:
            question_concat_basic_embedding=tf.concat([word_level_max_pooling_pq,word_level_avg_pooling_pq],2)
            choice_concat_basic_embedding=tf.concat([word_level_max_pooling_pc,word_level_avg_pooling_pc],2)
            qc_basic_dim=2
        qc_basic_embedding=my_rnn.concatenate_sents(question_concat_basic_embedding,choice_concat_basic_embedding, concat_idx_mat)# [batch_size, qc_len, qc_basic_dim]
        matching_tensors.append(question_concat_basic_embedding)
        matching_tensors.append(choice_concat_basic_embedding)
        matching_tensors.append(qc_basic_embedding)

    if construct_memory:
        memory=Memory(passage_lengths, tiled_memory_mask=tiled_mask, cond_training=cond_training)
        memory.add_memory_repre(untiled_word_level_max_pooling_qp,1)
        memory.add_memory_repre(untiled_word_level_avg_pooling_qp,1)
        memory.add_memory_repre(untiled_word_level_max_attentive_qp,MP_dim)


    all_match_templates=[]
    for matchid in rl_matches:
        if matchid in [1,2]:
            matcher=Matcher(matchid, tiled_question_lengths, choice_lengths, cond_training=cond_training)
            matcher.add_question_repre(word_level_max_pooling_pq,1)
            matcher.add_question_repre(word_level_avg_pooling_pq,1)
            matcher.add_choice_repre(word_level_max_pooling_pc,1)
            matcher.add_choice_repre(word_level_avg_pooling_pc,1)

            if MP_dim>0 and with_max_attentive_match:
                matcher.add_question_repre(word_level_max_attentive_pq, MP_dim)
                matcher.add_choice_repre(word_level_max_attentive_pc,MP_dim)

            all_match_templates.append(matcher)
        elif matchid==0:
            matcher=Matcher(matchid, tiled_question_lengths, None, cond_training=cond_training, qc_lengths=qc_lengths)
            matcher.add_question_repre(qc_basic_embedding,qc_basic_dim)
            all_match_templates.append(matcher)

    in_qc_repres = my_rnn.concatenate_sents(tiled_in_question_repres, in_choice_repres, concat_idx_mat)
    gate_input = []

    # print('here')
    with tf.variable_scope('context_MP_matching'):
        for i in range(context_layer_num): # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                with tf.variable_scope('context_represent'):
                    # parameters
                    context_lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    context_lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(context_lstm_dim)
                    if cond_training:
                        context_lstm_cell_fw=SwitchableDropoutWrapper(context_lstm_cell_fw, is_training, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw=SwitchableDropoutWrapper(context_lstm_cell_bw, is_training, output_keep_prob=(1 - dropout_rate))
                    elif is_training:
                        context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
                        context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_fw])
                    context_lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([context_lstm_cell_bw])





                    if concat_context:

                        print('concat context')


                        # question representation
                        (qc_context_representation_fw, qc_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_qc_repres, dtype=tf.float32, 
                                            sequence_length=qc_lengths) # [batch_size, question_len, context_lstm_dim]
                        in_qc_repres = tf.concat([qc_context_representation_fw, qc_context_representation_bw], 2)

                        # gate_input=my_rnn.extract_question_repre(qc_context_representation_fw,qc_context_representation_bw,question_lengths)
                        tiled_question_context_representation_fw, choice_context_representation_fw = \
                            my_rnn.split_sents(qc_context_representation_fw, split_idx_mat_q, split_idx_mat_c)
                        tiled_question_context_representation_bw, choice_context_representation_bw = \
                            my_rnn.split_sents(qc_context_representation_bw, split_idx_mat_q, split_idx_mat_c)

                        tiled_in_question_repres = tf.concat(
                            [tiled_question_context_representation_fw, tiled_question_context_representation_bw], 2)
                        in_choice_repres = tf.concat(
                            [choice_context_representation_fw, choice_context_representation_bw], 2)

                        # question representation
                        tf.get_variable_scope().reuse_variables()
                        (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32,
                                            sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                        in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)
                        gate_input.append(question_context_representation_fw[:,-1,:])
                        gate_input.append(question_context_representation_bw[:,0,:])


                        # Passage representation
                        tf.get_variable_scope().reuse_variables()
                        (passage_context_representation_fw, passage_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_passage_repres, dtype=tf.float32, 
                                            sequence_length=passage_lengths) # [batch_size, passage_len, context_lstm_dim]
                        in_passage_repres = tf.concat([passage_context_representation_fw, passage_context_representation_bw], 2)
                    else:
                        print('not concat context')
                        # question representation
                        (question_context_representation_fw, question_context_representation_bw), _ = my_rnn.bidirectional_dynamic_rnn(
                                            context_lstm_cell_fw, context_lstm_cell_bw, in_question_repres, dtype=tf.float32, 
                                            sequence_length=question_lengths) # [batch_size, question_len, context_lstm_dim]
                        in_question_repres = tf.concat([question_context_representation_fw, question_context_representation_bw], 2)

                        # gate_input=[question_context_representation_fw[:,-1,:],question_context_representation_bw[:,0,:]]
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

                        tiled_in_question_repres=maybe_tile(in_question_repres,efficient)
                        tiled_question_context_representation_fw=maybe_tile(question_context_representation_fw,efficient)
                        tiled_question_context_representation_bw=maybe_tile(question_context_representation_bw,efficient)

                        if 0 in rl_matches:
                            qc_context_representation_fw = my_rnn.concatenate_sents(tiled_question_context_representation_fw,
                                choice_context_representation_fw, concat_idx_mat)
                            qc_context_representation_bw = my_rnn.concatenate_sents(tiled_question_context_representation_bw,
                                choice_context_representation_bw, concat_idx_mat)
                            # in_qc_repres=my_rnn.concatenate_sents(in_question_repres,in_choice_repres,concat_idx_mat)
                if 0 in rl_matches:
                    qc_shape=tf.shape(qc_context_representation_fw)
                    qc_len=qc_shape[1]
                    qc_mask = tf.sequence_mask(qc_lengths, qc_len, dtype=tf.float32)                    
                matching_tensors.append(question_context_representation_fw)
                matching_tensors.append(question_context_representation_bw)
                matching_tensors.append(choice_context_representation_fw)
                matching_tensors.append(choice_context_representation_fw)
                matching_tensors.append(qc_context_representation_fw)
                matching_tensors.append(qc_context_representation_bw)

                tiled_passage_context_representation_fw=maybe_tile(passage_context_representation_fw,efficient)
                tiled_passage_context_representation_bw=maybe_tile(passage_context_representation_bw,efficient)


                # gate_input=tf.concat([tiled_question_context_representation_fw[:,-1,:],tiled_question_context_representation_bw[:,0,:]],1, name='gate_input')



                matched=False
                if 0 in rl_matches:
                    scope_name='first_matching' if tied_match else 'p_qc_matching'
                    reuse=True if tied_match and matched else None
                    matched=True
                    with tf.variable_scope(scope_name,reuse=reuse):
                        (p_qc_matching_vectors,p_qc_matching_dim) = match_passage_with_question(
                                qc_context_representation_fw, qc_context_representation_bw,qc_mask,
                                tiled_passage_context_representation_fw, tiled_passage_context_representation_bw, tiled_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                if 1 in rl_matches or 2 in rl_matches:
                    scope_name='first_matching' if tied_match else 'p_c_matching'
                    reuse=True if tied_match and matched else None
                    matched=True
                    with tf.variable_scope(scope_name,reuse=reuse):
                        (p_c_matching_vectors_fw, p_c_matching_vectors_bw, p_c_matching_dim_fw, p_c_matching_dim_bw) = \
                                match_passage_with_question_direct(
                                choice_context_representation_fw, choice_context_representation_bw,choice_mask,
                                tiled_passage_context_representation_fw, tiled_passage_context_representation_bw, tiled_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                with_direction=True)
                if 2 in rl_matches:
                    scope_name='first_matching' if tied_match else 'p_q_matching'
                    reuse=True if tied_match and matched else None
                    matched=True
                    if concat_context:
                        with tf.variable_scope(scope_name,reuse=reuse):
                            (p_q_matching_vectors_fw, p_q_matching_vectors_bw, p_q_matching_dim_fw, p_q_matching_dim_bw) = \
                                    match_passage_with_question_direct(
                                    tiled_question_context_representation_fw, tiled_question_context_representation_bw,tiled_question_mask,
                                    tiled_passage_context_representation_fw, tiled_passage_context_representation_bw, tiled_mask,
                                    MP_dim, context_lstm_dim, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                    with_direction=True)
                        p_q_matching_vectors_fw_concat = tf.concat(p_q_matching_vectors_fw, 2)
                        p_q_matching_vectors_bw_concat = tf.concat(p_q_matching_vectors_bw, 2)
                    else:
                        with tf.variable_scope(scope_name,reuse=reuse):
                            (p_q_matching_vectors_fw, p_q_matching_vectors_bw, p_q_matching_dim_fw, p_q_matching_dim_bw) = \
                                    match_passage_with_question_direct(
                                    question_context_representation_fw, question_context_representation_bw,question_mask,
                                    passage_context_representation_fw, passage_context_representation_bw, mask,
                                    MP_dim, context_lstm_dim, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match,
                                    with_direction=True)
                        p_q_matching_vectors_fw_concat = maybe_tile(tf.concat(p_q_matching_vectors_fw, 2),efficient)
                        p_q_matching_vectors_bw_concat = maybe_tile(tf.concat(p_q_matching_vectors_bw, 2),efficient)

                    # Multi-perspective matching
                    p_c_matching_vectors_fw_concat=tf.concat(p_c_matching_vectors_fw,2)
                    p_c_matching_vectors_bw_concat=tf.concat(p_c_matching_vectors_bw,2)
                    scope_name='second_matching' if tied_match else 'cq_post_MP_matching'
                    with tf.variable_scope(scope_name):
                        (c_q_postmatching_vectors, c_q_postmatching_dim) = match_passage_with_question_direct(
                                    p_q_matching_vectors_fw_concat, p_q_matching_vectors_bw_concat, tiled_question_mask,
                                    p_c_matching_vectors_fw_concat, p_c_matching_vectors_bw_concat, choice_mask,                                    
                                    MP_dim, p_q_matching_dim_fw, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    scope_name = 'second_matching' if tied_match else 'qc_post_MP_matching'
                    reuse = True if tied_match else None
                    with tf.variable_scope(scope_name,reuse=reuse):
                        (q_c_postmatching_vectors, q_c_postmatching_dim) = match_passage_with_question_direct(
                                    p_c_matching_vectors_fw_concat, p_c_matching_vectors_bw_concat, choice_mask,
                                    p_q_matching_vectors_fw_concat, p_q_matching_vectors_bw_concat, tiled_question_mask,
                                    MP_dim, p_q_matching_dim_fw, scope=None,
                                    with_full_match=with_full_match, with_maxpool_match=with_maxpool_match, 
                                    with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                aggre_defined=False
                if construct_memory:
                    scope_name='first_matching' if tied_match else 'q_p_matching'
                    reuse=True if tied_match and matched else None
                    matched=True
                    with tf.variable_scope(scope_name,reuse=reuse):
                        (q_p_matching_vectors, q_p_matching_dim) = match_passage_with_question(
                                passage_context_representation_fw, passage_context_representation_bw, mask,
                                question_context_representation_fw, question_context_representation_bw, question_mask,
                                MP_dim, context_lstm_dim, scope=None,
                                with_full_match=with_full_match, with_maxpool_match=with_maxpool_match,
                                with_attentive_match=with_attentive_match, with_max_attentive_match=with_max_attentive_match)
                    memory.add_memory_repre(q_p_matching_vectors,q_p_matching_dim,extend=True)
                for (mat_id,rl_match_opt) in enumerate(rl_matches):
                    current_matcher=all_match_templates[mat_id]
                    if rl_match_opt==0:
                        current_matcher.add_question_repre(p_qc_matching_vectors,p_qc_matching_dim,extend=True)
                    if rl_match_opt==1:
                        current_matcher.add_question_repre(tiled_question_context_representation_fw, context_lstm_dim)
                        current_matcher.add_question_repre(tiled_question_context_representation_bw, context_lstm_dim)
                        if 2 in rl_matches:
                            current_matcher.add_choice_repre(p_c_matching_vectors_fw_concat, p_c_matching_dim_fw)
                            current_matcher.add_choice_repre(p_c_matching_vectors_bw_concat, p_c_matching_dim_bw)
                        else:
                            current_matcher.add_choice_repre(p_c_matching_vectors_fw, p_c_matching_dim_fw,extend=True)
                            current_matcher.add_choice_repre(p_c_matching_vectors_bw, p_c_matching_dim_bw,extend=True)
                    if rl_match_opt==2:
                        current_matcher.add_question_repre(c_q_postmatching_vectors,c_q_postmatching_dim, extend=True)
                        current_matcher.add_choice_repre(q_c_postmatching_vectors,q_c_postmatching_dim, extend=True)
    gate_input=tf.concat(gate_input,1, name='gate_input')
    added_agg_highway=False
    matching_dim=all_match_templates[0].question_repre_dim
    reuse_choice_highway=None
    for mid,matcher in enumerate(all_match_templates):
        # matching_tensors.extend(matcher.question_repre)
        # matching_tensors.extend(matcher.choice_repre)
        matcher.concat(is_training,dropout_rate)
        if rl_matches[mid]==1 and tied_aggre:
            target_shape=[-1,question_len, matching_dim]
            transform_W=tf.get_variable('tied_aggre_fitting_lntransform',[matcher.question_repre_dim, matching_dim],tf.float32)
            matcher.transform_question_repre(transform_W,target_shape, matching_dim)

        if matcher.question_repre_dim>0:
            matching_tensors.append(matcher.question_repre)
        if matcher.choice_repre_dim>0:
            matching_tensors.append(matcher.choice_repre)

        reuse = True if mid>0 and tied_aggre else None

        if with_match_highway:
            print('adding matcher ',mid, 'matching highway: dim=', matcher.question_repre_dim, matcher.choice_repre_dim)
            matcher.add_highway_layer(highway_layer_num, tied_aggre=tied_aggre, reuse_question=reuse, reuse_choice=reuse_choice_highway)
            if rl_matches[mid]!=0 and tied_aggre:
                reuse_choice_highway=True
        agg_dim=matcher.aggregate(aggregation_layer_num, aggregation_lstm_dim, is_training, dropout_rate, tied_aggre=tied_aggre, reuse=reuse)
        print('aggregation dim=',agg_dim)
        if with_aggregation_highway:
            if not added_agg_highway:
                matcher.add_aggregation_highway(highway_layer_num, tied_aggre=tied_aggre, reuse=None)
                added_agg_highway=True
            else:
                matcher.add_aggregation_highway(highway_layer_num, tied_aggre=tied_aggre, reuse=True)
    if construct_memory:
        memory.concat(is_training, dropout_rate)
        matching_tensors.append(memory.question_repre)
        reuse=True if tied_aggre else None
        if with_match_highway:
            memory.add_highway_layer(highway_layer_num, tied_aggre=tied_aggre, reuse_question=reuse)
        memory.aggregate(aggregation_layer_num, aggregation_lstm_dim, is_training, dropout_rate, tied_aggre=tied_aggre, reuse=reuse)

    ret_list=[all_match_templates, agg_dim, gate_input]
    if construct_memory:
        ret_list.append(memory)
        # ret_list.append(tiled_mask)
    if debug:
        ret_list.append(matching_tensors)
    return ret_list