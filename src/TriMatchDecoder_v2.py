# -*- coding: utf-8 -*-

import argparse
from vocab_utils import Vocab
import namespace_utils

import tensorflow as tf
import TriMatchTrainer_v2 as TriMatchTrainer
from TriMatchModelGraph_v2 import TriMatchModelGraph
import sys

tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL
num_options=4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, required=True, help='word embedding file for the input file.')
    parser.add_argument('--mode', type=str, default="prediction", help='prediction or probs')
    parser.add_argument('--batch_size', default=None, help='test batch size')



    args, unparsed = parser.parse_known_args()
    
    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path
    word_vec_path = args.word_vec_path
    mode = args.mode
    out_json_path = None
    dump_prob_path = None

    # load the configuration file
    print('Loading configurations.')
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    print(FLAGS)

    with_POS=False
    if hasattr(FLAGS, 'with_POS'): with_POS = FLAGS.with_POS
    with_NER=False
    if hasattr(FLAGS, 'with_NER'): with_NER = FLAGS.with_NER
    wo_char = False
    if hasattr(FLAGS, 'wo_char'): wo_char = FLAGS.wo_char

    wo_left_match = False
    if hasattr(FLAGS, 'wo_left_match'): wo_left_match = FLAGS.wo_left_match

    wo_right_match = False
    if hasattr(FLAGS, 'wo_right_match'): wo_right_match = FLAGS.wo_right_match

    wo_full_match = False
    if hasattr(FLAGS, 'wo_full_match'): wo_full_match = FLAGS.wo_full_match

    wo_maxpool_match = False
    if hasattr(FLAGS, 'wo_maxpool_match'): wo_maxpool_match = FLAGS.wo_maxpool_match

    wo_attentive_match = False
    if hasattr(FLAGS, 'wo_attentive_match'): wo_attentive_match = FLAGS.wo_attentive_match

    wo_max_attentive_match = False
    if hasattr(FLAGS, 'wo_max_attentive_match'): wo_max_attentive_match = FLAGS.wo_max_attentive_match

    max_hyp_length = 100
    if hasattr(FLAGS, 'max_hyp_length'): max_hyp_length = FLAGS.max_hyp_length

    max_choice_length=None
    if hasattr(FLAGS, 'max_choice_length'): max_choice_length = FLAGS.max_choice_length

    matching_option=0
    if hasattr(FLAGS,'matching_option'): matching_option=FLAGS.matching_option

    use_options=False
    if hasattr(FLAGS,'use_options'): use_options=FLAGS.use_options

    cond_training=False
    if hasattr(FLAGS,'cond_training') or FLAGS.matching_option==7:
        cond_training=True

    if args.batch_size is not None:
        FLAGS.batch_size=args.batch_size
    gen_concat_mat=False
    gen_split_mat=False
    if FLAGS.matching_option==7:
        gen_concat_mat=True
        FLAGS.cond_training=True
        if FLAGS.concat_context:
            gen_split_mat=True

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(word_vec_path, fileformat='txt3',tolower=FLAGS.use_lower_letter)
    label_vocab = Vocab(model_prefix + ".label_vocab", fileformat='txt2',tolower=FLAGS.use_lower_letter)
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    
    POS_vocab = None
    NER_vocab = None
    char_vocab = None
    if with_POS: POS_vocab = Vocab(model_prefix + ".POS_vocab", fileformat='txt2',tolower=FLAGS.use_lower_letter)
    if with_NER: NER_vocab = Vocab(model_prefix + ".NER_vocab", fileformat='txt2',tolower=FLAGS.use_lower_letter)
    char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
    print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    print('Build TriMatchDataStream ... ')
    testDataStream = TriMatchTrainer.TriMatchDataStream(in_path, word_vocab=word_vocab, char_vocab=char_vocab, 
                                              POS_vocab=POS_vocab, NER_vocab=NER_vocab, label_vocab=label_vocab, 
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=False, 
                                              max_char_per_word=FLAGS.max_char_per_word, 
                                              max_sent_length=FLAGS.max_sent_length, max_hyp_length=max_hyp_length, 
                                              max_choice_length=max_choice_length,tolower=FLAGS.use_lower_letter,
                                              gen_concat_mat=gen_concat_mat, gen_split_mat=gen_split_mat)
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

    if wo_char: char_vocab = None

    init_scale = 0.01
    best_path = model_prefix + ".best.model"
    print('Decoding on the test set:')
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph =  TriMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,POS_vocab=POS_vocab, NER_vocab=NER_vocab, 
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
                 use_options=use_options, num_options=num_options, with_no_match=FLAGS.with_no_match, matching_option=matching_option,
                 cond_training=cond_training)

#             saver = tf.train.Saver()
        # remove word _embedding
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)
                
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        step = 0
        saver.restore(sess, best_path)

        accuracy = TriMatchTrainer.evaluate(testDataStream, valid_graph, sess, outpath=out_path, label_vocab=label_vocab,mode=args.mode,
                                                 char_vocab=char_vocab, POS_vocab=POS_vocab, NER_vocab=NER_vocab, use_options=use_options, 
                                                 cond_training=cond_training)
        print("Accuracy for test set is %.2f" % accuracy)


