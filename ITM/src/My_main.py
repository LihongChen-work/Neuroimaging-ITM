# -*- coding: utf-8 -*-
import time
from Model import *
import sys
import indexdocument
import result_display
import os


# 放的是BTM 模型的参数
def usage():
    print("Training Usage: \
    btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n\
    \tK  int, number of topics, like 20\n \
    \tW  int, size of vocabulary\n \
    \talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0\n \
    \tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01\n \
    \tn_iter  int, number of iterations of Gibbs sampling\n \
    \tsave_step   int, steps to save the results\n \
    \tdocs_pt     string, path of training docs\n \
    \tmodel_dir   string, output directory")


def BTM(argvs):
    if (len(argvs) < 4):
        usage()
    else:
        if (argvs[0] == "est"):
            K = argvs[1]
            W = argvs[2]
            alpha = argvs[3]
            beta = argvs[4]
            n_iter = argvs[5]
            save_step = argvs[6]
            docs_pt = argvs[7]
            dir = argvs[8]
            filename = argvs[9]
            print("===== Run My_BTM, K=" + str(K) + ", W=" + str(W) + ", alpha=" + str(alpha) + ", beta=" + str(
                beta) + ", n_iter=" + str(n_iter) + ", save_step=" + str(save_step) + "=====")
            # clock_start = time.pref_counter()
            model = Model(K, W, alpha, beta, n_iter, save_step)
            model.run(docs_pt, dir, filename, step=3)
            # clock_end = time.pref_counter()
            # print("procedure time : " + str(clock_end - clock_start))
        else:
            usage()


if __name__ == "__main__":
    mode = "est"
    K = 3
    W = None
    alpha = 0.5
    beta = 0.5
    n_iter = 10
    # 4次迭代
    save_step = 100
    dir = "../output/"
    input_dir = "../sample-data/"
    # model_dir = dir + "model/"  # 模型存放的文件夹
    # voca_pt = dir + "voca-id-word.txt"  # 生成的词典
    # dwid_pt = dir + "doc_wids.txt"  # 每篇文档由对应的序号单词组成

    # path = '../data_all_line_stopwords/'
    # path = '../data_abstract_line_stopwords/'
    # path = "../原始数据/txts-123/"
    path = "../原始数据/第1份实体对-2/"
    filesname = os.listdir(path)
    print(filesname)

    dir_dwid = "../output/model/dwid-ids/"
    dir_voca = "../output/model/voca-id-word/"

    # doc_pt = input_dir + "test.dat"  # 输入的文档

    print("=============== Index Docs =============")
    # W 生成的词典

    for filename in filesname:
        doc_pt = path + filename
        dwid_pt = dir_dwid + filename +"doc_wids.txt"
        voca_pt = dir_voca + filename
        model_dir = dir + "model/" + "pro/"+ filename

        # 返回的是词典中单词数
        W = indexdocument.run_indexDocs(['indexDocs', doc_pt, dwid_pt, voca_pt])

        print("W : " + str(W))  # 词典中单词的个数，数据写入../output/voca-id-word.txt文件

        argvs = []
        argvs.append(mode)
        argvs.append(K)
        argvs.append(W)
        argvs.append(alpha)
        argvs.append(beta)
        argvs.append(n_iter)
        argvs.append(save_step)
        argvs.append(dwid_pt)
        argvs.append(model_dir)
        argvs.append(filename)

        print("=============== Topic Learning =============")
        BTM(argvs)

        print("================ Topic Display =============")
        result_display.run_topicDicplay(['topicDisplay', model_dir, K, voca_pt], filename)


