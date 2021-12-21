import utils
import tf_utils
from process_data import process_data
import numpy as np
import tensorflow as tf
import sys
import os.path


def checkInputs():
    if (len(sys.argv) <= 3) or os.path.isfile(sys.argv[0])==False :
        raise ValueError(
            'The configuration file and the timestamp should be specified.')


if __name__ == "__main__":
    checkInputs()

    config = process_data(sys.argv[1])

    train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))  # np.arange
    dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
    test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))

    tf.reset_default_graph()  # 清除默认图形堆栈并重置全局默认图形
    tf.set_random_seed(1)  # 设置图级随机seed

    utils.printParameters(config)  # 打印配置
    with tf.Session() as sess:

        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                           trainable=False).assign(config.wordvectors)
        #
        saver = tf.train.import_meta_graph('./net/my_net.ckpt.meta')
        saver.restore(sess, "./net/my_net.ckpt")



        emb_mtx = sess.run(embedding_matrix)



        model = tf_utils.model(config, emb_mtx, sess)


        obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()

        train_step = model.get_train_op(obj)  #

        operations = tf_utils.operations(train_step, obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel,
                                         actual_op_rel, score_op_rel)  #

        sess.run(tf.global_variables_initializer())  # 初始化变量

        dev_score = model.evaluate(dev_data, operations, 'dev')



