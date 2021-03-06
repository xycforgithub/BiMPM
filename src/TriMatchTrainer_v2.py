# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import re
import tensorflow as tf

from vocab_utils import Vocab
from SentenceMatchDataStream import TriMatchDataStream
from TriMatchModelGraph_v2 import TriMatchModelGraph
import namespace_utils
import numpy as np
import json
import pickle

FLAGS = None
num_options=4
# tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL
def collect_vocabs(train_path, with_POS=False, with_NER=False,tolower=False):
    # Collect all vocabulary
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt',encoding='utf-8')
    for line in infile:
        line = line.strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        if tolower:
            sentence1 = re.split("\\s+",items[1].strip().lower())
            sentence2 = re.split("\\s+",items[2].strip().lower())
            sentence3 = re.split("\\s+",items[3].strip().lower())
        else:
            sentence1 = re.split("\\s+",items[1].strip())
            sentence2 = re.split("\\s+",items[2].strip())
            sentence3 = re.split("\\s+",items[3].strip())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        all_words.update(sentence3)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[4]))
            all_POSs.update(re.split("\\s+",items[5]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[6]))
            all_NERs.update(re.split("\\s+",items[7]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def evaluate(dataStream, valid_graph, sess, outpath=None, label_vocab=None, mode='prediction',
             char_vocab=None, POS_vocab=None, NER_vocab=None, use_options=False, cond_training=False,
             output_gate_probs=False, efficient=False):
    '''
    Evaluate all data in dataStream.
    '''
    if outpath is not None: outfile = open(outpath, 'wt',encoding='utf-8')
    # print('evaluate_v2')
    total_tags = 0.0
    correct_tags = 0.0
    dataStream.reset()
    correct_tag_list=[]
    total_count_list=[]
    if output_gate_probs:
        out_gate_path=outpath.replace('.probs','.gateprobs')
        print('gate_prob_path:',out_gate_path)
        out_gate_file=open(out_gate_path,'w')
    for batch_index in range(dataStream.get_num_batch()):
        if batch_index % 10 ==0:
            print(' %d/%d ' % (batch_index,dataStream.get_num_batch()), end="")
            sys.stdout.flush()
        cur_batch = dataStream.nextBatch()
        (label_batch, sent1_batch, sent2_batch, sent3_batch, label_id_batch,
                             word_idx_1_batch, word_idx_2_batch, word_idx_3_batch,
                             char_matrix_idx_1_batch, char_matrix_idx_2_batch, char_matrix_idx_3_batch, 
                             sent1_length_batch, sent2_length_batch, sent3_length_batch,
                             sent1_char_length_batch, sent2_char_length_batch, sent3_char_length_batch,
                             POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch,
                             concat_mat_batch, split_mat_batch_q, split_mat_batch_c) = cur_batch
        feed_dict = {
                     valid_graph.get_truth(): label_id_batch, 
                     valid_graph.get_passage_lengths(): sent1_length_batch, 
                     valid_graph.get_question_lengths(): sent2_length_batch,
                     valid_graph.get_choice_lengths(): sent3_length_batch, 
                     valid_graph.get_in_passage_words(): word_idx_1_batch, 
                     valid_graph.get_in_question_words(): word_idx_2_batch, 
                     valid_graph.get_in_choice_words(): word_idx_3_batch, 
                     # valid_graph.is_training: False
#                          valid_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                          valid_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                          valid_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                          valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                     }
        if cond_training:
            feed_dict[valid_graph.is_training]=False                    
        if char_vocab is not None:
            feed_dict[valid_graph.get_passage_char_lengths()] = sent1_char_length_batch
            feed_dict[valid_graph.get_question_char_lengths()] = sent2_char_length_batch
            feed_dict[valid_graph.get_choice_char_lengths()] = sent3_char_length_batch
            feed_dict[valid_graph.get_in_passage_chars()] = char_matrix_idx_1_batch
            feed_dict[valid_graph.get_in_question_chars()] = char_matrix_idx_2_batch
            feed_dict[valid_graph.get_in_choice_chars()] = char_matrix_idx_3_batch

        if POS_vocab is not None:
            feed_dict[valid_graph.get_in_passage_poss()] = POS_idx_1_batch
            feed_dict[valid_graph.get_in_question_poss()] = POS_idx_2_batch

        if NER_vocab is not None:
            feed_dict[valid_graph.get_in_passage_ners()] = NER_idx_1_batch
            feed_dict[valid_graph.get_in_question_ners()] = NER_idx_2_batch
        if concat_mat_batch is not None:
            feed_dict[valid_graph.concat_idx_mat] = concat_mat_batch
        if split_mat_batch_q is not None:
            feed_dict[valid_graph.split_idx_mat_q] = split_mat_batch_q
            feed_dict[valid_graph.split_idx_mat_c] = split_mat_batch_c

        total_tags += len(label_batch)
        to_eval=[valid_graph.get_eval_correct()]

        if outpath is not None:
            if mode == 'prediction':
                to_eval.append(valid_graph.get_predictions())
            else:
                to_eval.append(valid_graph.get_prob())
        if output_gate_probs:
            to_eval.append(valid_graph.final_log_probs)
        eval_res=sess.run(to_eval,feed_dict=feed_dict)

        correct_tag_list.append(int(eval_res[0]))
        # print('%d/%d'%(eval_res[0],len(label_batch)))
        total_count_list.append(len(label_batch))

        if use_options:
            correct_tags+=eval_res[0]*4
            # print('this correct tag=',eval_res[0])
        else:
            correct_tags += eval_res[0]
        if outpath is not None:
            if use_options:
                if mode=='prediction':
                    predictions = eval_res[1]
                    num_question=len(label_batch)//num_options
                    for i in range(num_question):
                        if efficient:
                            gt=str(np.argmax(label_id_batch[i::num_question]))
                        else:
                            gt=str(np.argmax(label_id_batch[i*4:(i+1)*4]))
                        outline=gt+"\t"+predictions[i]
                        outfile.write(outline)
                else:
                    probs = eval_res[1]
                    for i in range(len(label_batch)//num_options):
                        # import pdb
                        # pdb.set_trace()  
                        outfile.write(str(np.argmax(label_id_batch[i*4:(i+1)*4]))+"\t"+output_probs_options(probs[i]) + "\n")

            else:
                if mode =='prediction':
                    predictions = eval_res[1]
                    for i in range(len(label_batch)):
                        outline = label_batch[i] + "\t" + label_vocab.getWord(predictions[i]) + "\t" + sent1_batch[i] + "\t" + sent2_batch[i] + "\n"
                        # outfile.write(outline.encode('utf-8'))
                        outfile.write(outline)
                else:
                    probs = eval_res[1]
                    for i in range(len(label_batch)):
                        outfile.write(label_batch[i] + "\t" + output_probs(probs[i], label_vocab) + "\n")
        if output_gate_probs:
            gate_probs=np.transpose(np.exp(eval_res[2]))
            
            for i in range(len(label_batch)//num_options):
                out_gate_file.write(output_probs_options(gate_probs[i])+'\n')


    if outpath is not None: outfile.close()
    print('')
    # print(correct_tag_list,total_count_list)
    # json.dump({'correct':correct_tag_list,'total':total_count_list},open('res.txt','w'))
    accuracy = correct_tags / total_tags * 100
    return accuracy

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()
def output_probs_options(probs):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(i, probs[i])
    return out_string.strip()

def main(_):
    print('Configurations:')
    print(FLAGS)


    # Path and arguments configuraiton
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    test_path = FLAGS.test_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    tolower=FLAGS.use_lower_letter
    FLAGS.rl_matches=json.loads(FLAGS.rl_matches)
    gen_concat_mat=False
    gen_split_mat=False
    if FLAGS.matching_option==7:
        gen_concat_mat=True
        FLAGS.cond_training=True
        if FLAGS.concat_context:
            gen_split_mat=True
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    
    path_prefix = log_dir + "/TriMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    # namespace_utils.save_namespace(FLAGS, '../model_data/' + ".config.json")
    # input('check')
    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3',tolower=tolower)
    best_path = path_prefix + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    POS_path = path_prefix + ".POS_vocab"
    NER_path = path_prefix + ".NER_vocab"
    summary_path = path_prefix +'.summary_train'
    has_pre_trained_model = False
    POS_vocab = None
    NER_vocab = None


    print('best path:', best_path)
    if os.path.exists(best_path+'.data-00000-of-00001') and not(FLAGS.create_new_model):
        # Restore previously stored model
        print('Using pretrained model')
        has_pre_trained_model = True
        label_vocab = Vocab(label_path, fileformat='txt2',tolower=tolower)
        char_vocab = Vocab(char_path, fileformat='txt2',tolower=tolower)
        if FLAGS.with_POS: POS_vocab = Vocab(POS_path, fileformat='txt2',tolower=tolower)
        if FLAGS.with_NER: NER_vocab = Vocab(NER_path, fileformat='txt2',tolower=tolower)
    else:
        print('Creating new model')
        print('Collect words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path, 
                                                        with_POS=FLAGS.with_POS, with_NER=FLAGS.with_NER,tolower=tolower)
        if FLAGS.use_options:
            all_labels=['0','1']
        print('Number of words: {}'.format(len(all_words)))
        print('Number of labels: {}'.format(len(all_labels)))
        # for word in all_labels:
        #     print('label',word)
        # input('check')

        label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2,tolower=tolower)
        label_vocab.dump_to_txt2(label_path)

        print('Number of chars: {}'.format(len(all_chars)))
        char_vocab = Vocab(fileformat='voc', voc=all_chars,dim=FLAGS.char_emb_dim,tolower=tolower)
        char_vocab.dump_to_txt2(char_path)
        
        if FLAGS.with_POS:
            print('Number of POSs: {}'.format(len(all_POSs)))
            POS_vocab = Vocab(fileformat='voc', voc=all_POSs,dim=FLAGS.POS_dim,tolower=tolower)
            POS_vocab.dump_to_txt2(POS_path)
        if FLAGS.with_NER:
            print('Number of NERs: {}'.format(len(all_NERs)))
            NER_vocab = Vocab(fileformat='voc', voc=all_NERs,dim=FLAGS.NER_dim,tolower=tolower)
            NER_vocab.dump_to_txt2(NER_path)
            
    print('all_labels:',label_vocab)
    print('has pretrained model:',has_pre_trained_model)
    # for word in word_vocab.word_vecs:

    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    print('tag_vocab shape is {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    print('Build TriMatchDataStream ... ')

    #Build datastream

    trainDataStream = TriMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=True, isLoop=True, 
                                              isSort=(not FLAGS.wo_sort_instance_based_on_length), 
                                              max_char_per_word=FLAGS.max_char_per_word, 
                                              max_sent_length=FLAGS.max_sent_length,max_hyp_length=FLAGS.max_hyp_length, 
                                              max_choice_length=FLAGS.max_choice_length, tolower=tolower,
                                              gen_concat_mat=gen_concat_mat, gen_split_mat=gen_split_mat, efficient=FLAGS.efficient, 
                                              random_seed=FLAGS.random_seed)
                                    
    devDataStream = TriMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, 
                                              isSort=(not FLAGS.wo_sort_instance_based_on_length), 
                                              max_char_per_word=FLAGS.max_char_per_word, 
                                              max_sent_length=FLAGS.max_sent_length,max_hyp_length=FLAGS.max_hyp_length, 
                                              max_choice_length=FLAGS.max_choice_length, tolower=tolower,
                                              gen_concat_mat=gen_concat_mat, gen_split_mat=gen_split_mat, efficient=FLAGS.efficient, 
                                              random_seed=FLAGS.random_seed)

    testDataStream = TriMatchDataStream(test_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, 
                                              isSort=(not FLAGS.wo_sort_instance_based_on_length), 
                                              max_char_per_word=FLAGS.max_char_per_word, 
                                              max_sent_length=FLAGS.max_sent_length,max_hyp_length=FLAGS.max_hyp_length, 
                                              max_choice_length=FLAGS.max_choice_length, tolower=tolower,
                                              gen_concat_mat=gen_concat_mat, gen_split_mat=gen_split_mat, efficient=FLAGS.efficient, 
                                              random_seed=FLAGS.random_seed)

    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    
    sys.stdout.flush()
    if FLAGS.wo_char: char_vocab = None

    best_accuracy = 0.0
    init_scale = 0.01
    with tf.Graph().as_default():
        if FLAGS.random_seed is not None:
            tf.set_random_seed(FLAGS.random_seed)
            print('tf random seed set')
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
#         with tf.name_scope("Train"):
        # Build model
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = TriMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                 dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                 lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                 aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=True, MP_dim=FLAGS.MP_dim, 
                 context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                 fix_word_vec=FLAGS.fix_word_vec, with_highway=FLAGS.with_highway,
                 word_level_MP_dim=FLAGS.word_level_MP_dim,
                 with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                 highway_layer_num=FLAGS.highway_layer_num,
                 match_to_question=FLAGS.match_to_question, match_to_passage=FLAGS.match_to_passage, match_to_choice=FLAGS.match_to_choice,
                 with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                 with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match), 
                 use_options=FLAGS.use_options, num_options=num_options, with_no_match=FLAGS.with_no_match, verbose=FLAGS.verbose, 
                 matching_option=FLAGS.matching_option, concat_context=FLAGS.concat_context, 
                 tied_aggre=FLAGS.tied_aggre, rl_training_method=FLAGS.rl_training_method, rl_matches=FLAGS.rl_matches, 
                 cond_training=FLAGS.cond_training,reasonet_training=FLAGS.reasonet_training, reasonet_steps=FLAGS.reasonet_steps, 
                 reasonet_hidden_dim=FLAGS.reasonet_hidden_dim, reasonet_lambda=FLAGS.reasonet_lambda, 
                 reasonet_terminate_mode=FLAGS.reasonet_terminate_mode, reasonet_keep_first=FLAGS.reasonet_keep_first, 
                 efficient=FLAGS.efficient, tied_match=FLAGS.tied_match, reasonet_logit_combine=FLAGS.reasonet_logit_combine)


            tf.summary.scalar("Training Loss", train_graph.get_loss()) # Add a scalar summary for the snapshot loss.


        if FLAGS.cond_training:
            valid_graph=train_graph
            print('cond training')
        else:
            # in this case, build a another validation model for testing
            if FLAGS.verbose:
                valid_graph=train_graph
            else:
        #         with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    valid_graph = TriMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                         dropout_rate=0.0, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                         lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim, 
                         aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim, 
                         context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num, 
                         fix_word_vec=FLAGS.fix_word_vec, with_highway=FLAGS.with_highway,
                         word_level_MP_dim=FLAGS.word_level_MP_dim,
                         with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                         highway_layer_num=FLAGS.highway_layer_num,
                         match_to_question=FLAGS.match_to_question, match_to_passage=FLAGS.match_to_passage, match_to_choice=FLAGS.match_to_choice,
                         with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match), 
                         with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match), 
                         use_options=FLAGS.use_options, num_options=num_options, with_no_match=FLAGS.with_no_match,
                         matching_option=FLAGS.matching_option, concat_context=FLAGS.concat_context, 
                         tied_aggre=FLAGS.tied_aggre, rl_training_method=FLAGS.rl_training_method, rl_matches=FLAGS.rl_matches,
                         reasonet_training=FLAGS.reasonet_training, reasonet_steps=FLAGS.reasonet_steps,
                         reasonet_hidden_dim=FLAGS.reasonet_hidden_dim, reasonet_lambda=FLAGS.reasonet_lambda,
                         reasonet_terminate_mode=FLAGS.reasonet_terminate_mode, reasonet_keep_first=FLAGS.reasonet_keep_first, efficient=FLAGS.efficient,
                         tied_match=FLAGS.tied_match, reasonet_logit_combine=FLAGS.reasonet_logit_combine)

                
        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            # print(var.name,var.get_shape().as_list())
            if "word_embedding" in var.name: continue
            # print(var.name)
#             if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
        # input('check')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        sess = tf.Session(config=config)
        summary_writer= tf.summary.FileWriter(summary_path,sess.graph)
        print('log_directory:',summary_path)
        sess.run(initializer)
        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        sub_loss_counter=0.0
        graph_def=tf.get_default_graph().as_graph_def()
        # f.write()
        tensor_list=graph_def.node
        thenode=None
        attention_node_list=[]
        # for node in tensor_list:
        #     if ('reasonet_attention_softmax' in node.name) or ('test_argmax' in node.name):
        #         print(node.name)
        #         attention_node_list.append(tf.get_default_graph().get_tensor_by_name(node.name+':0'))

        for step in range(max_steps):
            # read data
            cur_batch = trainDataStream.nextBatch()
            (label_batch, sent1_batch, sent2_batch, sent3_batch, label_id_batch,
                                 word_idx_1_batch, word_idx_2_batch, word_idx_3_batch,
                                 char_matrix_idx_1_batch, char_matrix_idx_2_batch, char_matrix_idx_3_batch, 
                                 sent1_length_batch, sent2_length_batch, sent3_length_batch,
                                 sent1_char_length_batch, sent2_char_length_batch, sent3_char_length_batch,
                                 POS_idx_1_batch, POS_idx_2_batch, NER_idx_1_batch, NER_idx_2_batch,
                                 concat_mat_batch, split_mat_batch_q, split_mat_batch_c) = cur_batch

            # print(label_id_batch)
            if FLAGS.verbose:
                print(label_id_batch)
                print(sent1_length_batch)
                print(sent2_length_batch)
                print(sent3_length_batch)
                print(np.reshape(label_id_batch,[num_options,-1]))
                # print(word_idx_1_batch)
                # print(word_idx_2_batch)
                # print(word_idx_3_batch)
                # print(sent1_batch)
                # print(sent2_batch)
                # print(sent3_batch)
                # print(concat_mat_batch)
                # print(split_mat_batch_q)
                # print(split_mat_batch_c)
                input('check')
            feed_dict = {
                         train_graph.get_truth(): label_id_batch, 
                         train_graph.get_passage_lengths(): sent1_length_batch, 
                         train_graph.get_question_lengths(): sent2_length_batch,
                         train_graph.get_choice_lengths(): sent3_length_batch, 
                         train_graph.get_in_passage_words(): word_idx_1_batch, 
                         train_graph.get_in_question_words(): word_idx_2_batch, 
                         train_graph.get_in_choice_words(): word_idx_3_batch, 
#                          train_graph.get_question_char_lengths(): sent1_char_length_batch, 
#                          train_graph.get_passage_char_lengths(): sent2_char_length_batch, 
#                          train_graph.get_in_question_chars(): char_matrix_idx_1_batch, 
#                          train_graph.get_in_passage_chars(): char_matrix_idx_2_batch, 
                         }
            if FLAGS.cond_training:
                feed_dict[train_graph.is_training]=True
            if char_vocab is not None:
                feed_dict[train_graph.get_passage_char_lengths()] = sent1_char_length_batch
                feed_dict[train_graph.get_question_char_lengths()] = sent2_char_length_batch
                feed_dict[train_graph.get_choice_char_lengths()] = sent3_char_length_batch
                feed_dict[train_graph.get_in_passage_chars()] = char_matrix_idx_1_batch
                feed_dict[train_graph.get_in_question_chars()] = char_matrix_idx_2_batch
                feed_dict[train_graph.get_in_choice_chars()] = char_matrix_idx_3_batch

            if POS_vocab is not None:
                feed_dict[train_graph.get_in_passage_poss()] = POS_idx_1_batch
                feed_dict[train_graph.get_in_question_poss()] = POS_idx_2_batch

            if NER_vocab is not None:
                feed_dict[train_graph.get_in_passage_ners()] = NER_idx_1_batch
                feed_dict[train_graph.get_in_question_ners()] = NER_idx_2_batch
            if concat_mat_batch is not None:
                feed_dict[train_graph.concat_idx_mat] = concat_mat_batch
            if split_mat_batch_q is not None:
                feed_dict[train_graph.split_idx_mat_q] = split_mat_batch_q
                feed_dict[train_graph.split_idx_mat_c] = split_mat_batch_c

            if FLAGS.verbose:
                # Debug code
                # return_list = sess.run([train_graph.get_train_op(), train_graph.get_loss(), train_graph.get_predictions(),train_graph.get_prob(),
                    # train_graph.all_probs, train_graph.correct]+train_graph.matching_vectors, feed_dict=feed_dict)
                return_list = sess.run([train_graph.get_train_op(), train_graph.loss_summary, train_graph.get_loss(), train_graph.get_predictions(),train_graph.get_prob(),
                    train_graph.all_probs, train_graph.correct, train_graph.gate_prob, train_graph.gate_log_prob,
                    train_graph.weighted_log_probs, train_graph.log_coeffs, train_graph.gold_matrix, train_graph.final_log_probs]+train_graph.matching_vectors, feed_dict=feed_dict)

                print(len(return_list))
                with open('../model_data/res.pkg','wb') as fout:
                    pickle.dump(return_list, fout)
                input('written')
                _, loss_summary,loss_value, pred, prob, all_probs, correct, gate_prob, gate_log_prob, weighted_log_probs,\
                    log_coeffs,gold_matrix=return_list[0:12]
                print('loss=',loss_value) 
                print('pred=',pred)
                print('prob=',prob)
                print('all_probs=',all_probs)
                print('correct=',correct)
                print('gate_prob',gate_prob)
                print('gate_log_prob',gate_log_prob)
                print('weighted log probs=',weighted_log_probs)
                print('log_coeffs=',log_coeffs)
                print('gold_matrix=',gold_matrix)
                for val in return_list[10:]:
                    if isinstance(val,list):
                        print('list len ',len(val))
                        for objj in val:
                            print('this shape=',val.shape)
                    print('this shape=',val.shape)
                    # print(val)
                # print('question repre:',return_list[10][:,:,0])
                # print('choice repre:',return_list[12][:,:,0])
                # print('qc repre:',return_list[14][:,:,0])

                input('check')
            else:
                _, loss_summary, loss_value = sess.run([train_graph.get_train_op(), train_graph.loss_summary, train_graph.get_loss()], feed_dict=feed_dict)
            total_loss += loss_value
            summary_writer.add_summary(loss_summary, step)
            sub_loss_counter+=loss_value
            
            if step % int(FLAGS.display_every)==0: 
                print('{},{} '.format(step,sub_loss_counter), end="")
                sys.stdout.flush()
                sub_loss_counter=0.0

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps:
                print()
                # Print status to stdout.
                duration = time.time() - start_time
                start_time = time.time()
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                total_loss = 0.0

                # Evaluate against the validation set.
                print('Validation Data Eval:')
                if FLAGS.predict_val:
                    outpath=path_prefix+'.iter%d' % (step) +'.probs'
                else:
                    outpath=None
                # Test on validation data
                accuracy = evaluate(devDataStream, valid_graph, sess,char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                    use_options=FLAGS.use_options,outpath=outpath, mode='prob', cond_training=FLAGS.cond_training,efficient=FLAGS.efficient)
                print("Current accuracy on dev set is %.2f" % accuracy)
                if not FLAGS.not_save_model:
                    saver.save(sess, best_path+'_iter{}'.format(step))
                    print('saving the current model.')
                if accuracy>=best_accuracy:
                    best_accuracy = accuracy
                    if not FLAGS.not_save_model:
                        saver.save(sess, best_path)
                    print('saving the current model as best model.')
                if not FLAGS.not_save_model:
                    # Test on test data (just for reference)
                    accuracy = evaluate(testDataStream, valid_graph, sess,char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                        use_options=FLAGS.use_options,outpath=outpath, mode='prob', cond_training=FLAGS.cond_training,efficient=FLAGS.efficient)
                    print("Current accuracy on test set is %.2f" % accuracy)
                
    print("Best accuracy on dev set is %.2f" % best_accuracy)
    # decoding
    print('Decoding on the test set:')
    sess.close()
    init_scale = 0.01

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        print('current scope:',tf.get_variable_scope().name)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            print('current scope:',tf.get_variable_scope().name)

            valid_graph = TriMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
                         dropout_rate=0.0, learning_rate=FLAGS.learning_rate, optimize_type=FLAGS.optimize_type,
                         lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim, context_lstm_dim=FLAGS.context_lstm_dim,
                         aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False, MP_dim=FLAGS.MP_dim,
                         context_layer_num=FLAGS.context_layer_num, aggregation_layer_num=FLAGS.aggregation_layer_num,
                         fix_word_vec=FLAGS.fix_word_vec, with_highway=FLAGS.with_highway,
                         word_level_MP_dim=FLAGS.word_level_MP_dim,
                         with_match_highway=FLAGS.with_match_highway, with_aggregation_highway=FLAGS.with_aggregation_highway,
                         highway_layer_num=FLAGS.highway_layer_num,
                         match_to_question=FLAGS.match_to_question, match_to_passage=FLAGS.match_to_passage, match_to_choice=FLAGS.match_to_choice,
                         with_full_match=(not FLAGS.wo_full_match), with_maxpool_match=(not FLAGS.wo_maxpool_match),
                         with_attentive_match=(not FLAGS.wo_attentive_match), with_max_attentive_match=(not FLAGS.wo_max_attentive_match),
                         use_options=FLAGS.use_options, num_options=num_options, with_no_match=FLAGS.with_no_match, verbose=FLAGS.verbose,
                         matching_option=FLAGS.matching_option, concat_context=FLAGS.concat_context,
                         tied_aggre=FLAGS.tied_aggre, rl_training_method=FLAGS.rl_training_method, rl_matches=FLAGS.rl_matches,
                         cond_training=FLAGS.cond_training,reasonet_training=FLAGS.reasonet_training, reasonet_steps=FLAGS.reasonet_steps,
                         reasonet_hidden_dim=FLAGS.reasonet_hidden_dim, reasonet_lambda=FLAGS.reasonet_lambda,
                         reasonet_terminate_mode=FLAGS.reasonet_terminate_mode, reasonet_keep_first=FLAGS.reasonet_keep_first,
                         efficient=FLAGS.efficient, tied_match=FLAGS.tied_match, reasonet_logit_combine=FLAGS.reasonet_logit_combine)

        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
                
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)

        accuracy = evaluate(testDataStream, valid_graph, sess,
            char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, use_options=FLAGS.use_options, 
            cond_training=FLAGS.cond_training,efficient=FLAGS.efficient)
        print("Accuracy for test set is %.2f" % accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=100, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--MP_dim', type=int, default=10, help='Number of perspectives for matching vectors.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=100, help='Maximum number of words within each sentence.')
    parser.add_argument('--max_hyp_length', type=int, default=100, help='Maximum number of words within hypothesis.')
    parser.add_argument('--max_choice_length', default=None, help='Maximum number of words within choice.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='normal', required=True, help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=False, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_filter_layer', default=False, help='Utilize filter layer.', action='store_true')
    parser.add_argument('--word_level_MP_dim', type=int, default=-1, help='Number of perspectives for word-level matching.')
    parser.add_argument('--with_match_highway', default=False, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=False, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_POS', default=False, help='Utilize POS information.', action='store_true')
    parser.add_argument('--with_NER', default=False, help='Utilize NER information.', action='store_true')
    parser.add_argument('--POS_dim', type=int, default=20, help='Number of dimension for POS embeddings.')
    parser.add_argument('--NER_dim', type=int, default=20, help='Number of dimension for NER embeddings.')
    parser.add_argument('--match_to_passage', default=False, help='Match on passage encodings.', action='store_true')
    parser.add_argument('--match_to_question', default=False, help='Match on question encodings.', action='store_true')
    parser.add_argument('--match_to_choice', default=False, help='Match on choice encodings.', action='store_true')
    parser.add_argument('--wo_full_match', default=False, help='Without full matching.', action='store_true')
    parser.add_argument('--wo_maxpool_match', default=False, help='Without maxpooling matching', action='store_true')
    parser.add_argument('--wo_attentive_match', default=False, help='Without attentive matching', action='store_true')
    parser.add_argument('--wo_max_attentive_match', default=False, help='Without max attentive matching.', action='store_true')
    parser.add_argument('--wo_char', default=False, help='Without character-composed embeddings.', action='store_true')
    parser.add_argument('--use_options',default=False, help='Use softmax on RACE options',action='store_true')
    parser.add_argument('--verbose',default=False, help='Print test information',action='store_true')
    parser.add_argument('--wo_sort_instance_based_on_length',default=False,help='Without sorting sentences based on length',action='store_true')
    parser.add_argument('--with_no_match',default=False,help='Does not perform any matching',action='store_true')
    parser.add_argument('--display_every',default=100,help='Display progress every X step.')
    parser.add_argument('--use_lower_letter',default=False,help='Convert all words to lower case.')
    parser.add_argument('--predict_val',default=False,help='Give probs to dev set after each epoch.',action='store_true')
    parser.add_argument('--matching_option',type=int,default=0,help='TriMatch Configuration.')
    parser.add_argument('--create_new_model',default=False,help='Create new model regardless of the old one.',action='store_true')
    parser.add_argument('--concat_context', default=False, help='Concat question & choice and feed into context LSTM.', action='store_true')
    parser.add_argument('--tied_aggre', default=False,help='Tie aggregation layer weights.', action='store_true')
    parser.add_argument('--tied_match',default=False,help='Tie matching weights across different matchers.',action='store_true')
    parser.add_argument('--rl_training_method', default='contrastive', help='Method of RL to train gate.')
    parser.add_argument('--rl_matches', default='[0,1,2]', help='list of RL matcher templates.')
    parser.add_argument('--cond_training', default=False, help='Construct a graph conditional on is_training sign.', action='store_true')
    parser.add_argument('--efficient',default=False, help='Improve efficiency by processing passage only once.', action='store_true')
    parser.add_argument('--reasonet_training',default=False, help='Apply reasonet module', action='store_true')
    parser.add_argument('--reasonet_steps',type=int, default=5, help='Reasonet reading steps')
    parser.add_argument('--reasonet_hidden_dim',type=int,default=128, help='Reasonet hidden dimension (map both state and memory to this dimension).')
    parser.add_argument('--reasonet_lambda',type=int,default=10, help='multiplier in reasonet to enlarge differences')
    parser.add_argument('--reasonet_terminate_mode',default='original', help='reasonet terminate mode: original to use terminate probs, softmax to use terminate logits')
    parser.add_argument('--reasonet_keep_first', default=False, help='Also use step 0 as a gate in reasonet', action='store_true')
    parser.add_argument('--reasonet_logit_combine',default='sum', help='Use sum/max pooling to combine terminate logit of different answers.')
    parser.add_argument('--not_save_model',default=False,help='whether to save models (only use for testing).',action='store_true')
    parser.add_argument('--random_seed',type=int,default=None, help='random seed for graph init and data shuffle.')
#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

