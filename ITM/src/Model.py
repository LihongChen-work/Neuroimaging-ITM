# -*- coding: utf-8 -*-
from pvec import *
import numpy as np
from document import Doc
from samp import *


# ================================================================================================================
# 生成模型需要三个关键的函数，分别是load_docs()，model_init(),update_biterm()，就是通过这三个函数，我们生成了BTM模型

class Model():
    '''
    @description: 函数的功能是生成模型
    @param 
    @return: 
    '''
    W = 0   # vocabulary size 词汇量
    K = 0   # number of topics 主题数量
    n_iter = 0  # maximum number of iteration of Gibbs Sampling，吉布斯采样的最大迭代次数。
    save_step = 0
    alpha = 0   # hyperparameters of p(z)
    beta = 0    # hyperparameters of p(w|z)
    # Pvec()这个类是相当于作者自己实现的一个numpy
    nb_z = Pvec()   # n(b|z), size K*1 所有的词对biterm分配到主题z上的次数,在论文中是用来计算Nz的
    nwz = np.zeros((1, 1))  # n(w,z), size K*W，表示的是单词分配到主题z上的次数.array([[0, 0]]),即一行一列元素为0的数组
    pw_b = Pvec()   # 文档中每个词出现的频率，即 词：词频率；这三个参数都是为了计算每个主题中单词的分布而存在的。

    nw1_step1 = 0
    nw1_step2 = 0  # 实体1分配到主题z上，且在步长为2的关系中的次数
    nw1_step3 = 0
    nw2_step1 = 0
    nw2_step2 = 0  # 实体2分配到主题z上，且在步长为2的关系中的次数
    nw2_step3 = 0
    # bs = []   # 局部变量


    '''
        If true, the topic 0 is set to a background topic that 
        equals to the empirical word distribution. It can filter out common words
        如果为true，则主题0设置为等于经验单词分布的背景主题。它能过滤掉常见的单词
    '''
    has_background = False

    # __init__函数的参数列表会在开头多出一项，它永远指代新建的那个实例对象
    def __init__(self, K, W, a, b, n_iter, save_step, has_b=False):
        '''
        @description:初始化参数
        @param docs_pt:
        @return: 无
        '''
        self.K = K
        self.W = W
        self.alpha = a
        self.beta = b
        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b
        self.pw_b.resize(W)
        self.nwz.resize((K, W))  # 单词主题矩阵，表示的是单词分配到主题z上的次数，即K行W列的数组
        self.nb_z.resize(K)  # 主题集合，表示的是biterm分配到主题z上的次数



    # self.bi即bi在本类中共享，为实例变量
    def run(self, doc_pt, res_dir, filename, step):
        '''
        @description: 生成模型运行函数，得到的p(w|z)、p(z)会被写入model文件夹中（狄利克雷-多项式的共轭分布，Gibbs采样）
        @param {type} 
        @return: 
        '''
        bs = self.load_docs(doc_pt)  # 此处的bi正确
        """
        load_docs(doc_pt)函数的功能：
        # 生成self.pw_b 和 self.bs。 
        # 目前self.pw_b表示的是每个单词对应的词频，一共有7个单词，那么pw_b的size就是7
          self.bs表示的是所有的biterm
        """
        # print('=====单词在整个词库中的词频、词库中单词的总个数、词对集中所有biterm的编号=====')
        # print(self.pw_b.size_w())
        # for item in self.bs:  # 遍历biterm集合中的所有元素，并把单个biterm中的两个词语编号打印出来
        #     print(item.get_wi(), item.get_wj())
        # print('==========')
        # 以上代码是用来初始化self.pw_b 和 self.bs


        nw1, nw2 = self.model_init(filename, bs, step)  # 给出了bs
        """
        model_init()函数的功能：
        # 初始化 self.nb_z 和 self.nwz  
          self.nb_z: 表示的是在那么多的词对里面，每个主题出现的次数。
          self.nwz[2][3] 表示的是在主题2中，2号单词出现的次数。
        """
        # print('=====一个列表为某个主题中各个单词出现的次数、主题数=====')
        # print(self.nwz)
        # print(self.nb_z.size_b())  # 打印出每个主题中的biterm个数，并返回主题的个数
        # print('=======================')


        print('\n')
        print("Begin iteration 开始迭代")  # 调用吉布斯采样函数
        out_dir = res_dir + "k" + str(self.K) + "."  # 程序运行时数据p(w|z)、p(z)会被写入model文件夹中，
                                                     # 并且当主题数为3时，会生成文件k3.pw_z和k3.pz
        for i in range(1, self.n_iter+1):
            print("\riter "+str(i)+"/"+str(self.n_iter), end='\r')  # 此处输出的是迭代次数
            for b in range(len(bs)):

                # 根据每个biterm更新文章中的参数。
                self.update_biterm(bs[b], filename, bs, step, nw1, nw2)  # 计算核心代码，self.bs中保存的是词对的biterm类，代码是对每一个词对进行更新的。
            if i % self.save_step == 0:
                self.save_res(out_dir)

        self.save_res(out_dir)



    def model_init(self, filename, bs, step):
        '''
        @description: 初始化模型的代码，先初始化biterm主题，然后统计在bs中各个主题出现的次数(每个主题中biterm出现的次数)、每个主题中单词出现的次数
        @param :None
        @return: 生成self.nv_z 和self.nwz，
        @self.nb_z: 表示某主题下包含的biterm的个数。self.nv_z[1]:表示的第一个主题出现的次数。
        @self.nwz[2][3] 表示的是在主题2中，2号单词出现的次数。即表示某主题下包含的单词的个数
        '''
        nw_all_e1 = 0
        nw_all_e2 = 0

        for biterm in bs:
            # print('==============显示所有的biterm===============')
            # print(biterm.get_wi(), biterm.get_wj())

            # 函数uni_sample()用来生成随机数的，使用对每个起始的主题初始化个数
            k = uni_sample(self.K)  # k表示的是从0-K之间的随机数。用来初始化词对的主题
            # print('该biterm被初始化的主题类型序号是：', k)
            """
            # 函数assign_biterm_topic()用来初始化self.nb_z和self.nwz
            # self.nbz表示的是每个主题出现的次数,比如self.nb_z[1] = 3 表示第一个主题出现3次
            # self.nwz是一个二维数组，表示的是每个主题中，各个单词出现的次数。
              self.nwz[2][3]表示的是在2号主题中，3号单词出现的次数
            """
            # 输入参数是一个biterm和该biterm所初始化的主题类型，即先给biterm指定一个主题，经过吉布斯采样后才会得到它真正的主题类型
            # 在主题模型中主题的类型是用数字表示的
            nw_e1, nw_e2 = self.assign_biterm_topic(biterm, k, filename, step)
            nw_all_e1 = nw_e1 + nw_all_e1
            nw_all_e2 = nw_e2 + nw_all_e2
            # print('\n')
        print("实体1总的是次数： ", nw_all_e1)
        print("实体2总的是次数： ", nw_all_e2)
        return nw_all_e1, nw_all_e2

    # 由 “实体,实体” 组成的输入数据，# 此处的bi正确，bs的源头就是该函数
    def load_docs(self, docs_pt):
        '''
        @description: 读取文档，并生成self.pw_b（文档中每个词出现的频率，即 词：词频率 ）和 self.bs（所有的biterm对）
        @param docs_pt:
        @return: 无
        '''
        bs = []
        print("load docs: " + docs_pt)
        rf = open(docs_pt)
        if not rf:
            print("file not found: " + docs_pt)

        for line in rf.readlines():  # 读取的是dwid-ids文件
            d = Doc(line)
            biterms = []  # 一句话里的单词能组成的词对。
            d.gen_biterms(biterms)

            # statistic the empirical word distribution

            # ====================================
            for i in range(d.size()):
                w = d.get_w(i)  # 取到原始数据一行中的实体
                self.pw_b[w] += 1  # 这行代码是在统计词频
            for b in biterms:
                bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有的关系二元组.

        self.pw_b.normalize()  # 做归一化处理,现在 pw_b中保存的是 词：词频。

        """
        pw_b中保存的是 词：词频率
        """
        return bs



    # ============================================================================================================
    # Gibbs采样的部分,单纯地调用了4个函数（不需要修改！！！）
    def update_biterm(self, bi, filename, bs, step, nw1, nw2):
        self.reset_biterm_topic(bi)  # 排除当前biterm的主题；bi是某个biterm
        # comput p(z|b),相当于论文中计算Zb
        pz = Pvec()
        self.comput_pz_b(bi, pz, bs, nw1, nw2)  # 计算出来的结果，直接作用在pz上。
        # print(pz.size())  # pz是一个三个具体的数，如果主题的个数是5的话，那么pz就是5个具体的数。
        # print(pz.to_vector())  # pz.to_vector()表示将三个数转成向量。
        # sample topic for biterm b
        k = mul_sample(pz.to_vector())  # k表示根据pz算出三个数中最大的主题
        # print(k)
        # print('-----------')
        # print('\n')
        self.assign_biterm_topic(bi, k, filename, step)  # 更新论文中的Nz,N_wiz,N_wjz.


    # 该函数的功能：排除当前的biterm
    def reset_biterm_topic(self, bi):
        k = bi.get_z()
        w1 = bi.get_wi()
        w2 = bi.get_wj()
        self.nb_z[k] -= 1
        self.nwz[k][w1] -= 1
        self.nwz[k][w2] -= 1
        assert(self.nb_z[k] > -10e-7 and self.nwz[k][w1] > -10e-7 and self.nwz[k][w2] > -10e-7)
        bi.reset_z()


    # 重要函数！！！此处的bi不对*****************************************************
    # 该函数的功能是：更新统计在bs中各个主题出现的次数(每个主题中biterm出现的次数)、每个主题中单词出现的次数
    def assign_biterm_topic(self, bi, k, filename, step):
        # 需要判断关系二元组的步长，以确定权重
        # bi是每一个词对，k是主题的个数。

        bi.set_z(k)
        w1 = bi.get_wi()  # 词对中的第一个词
        w2 = bi.get_wj()  # 词对中的第二个词
        print("w1: ", w1)
        print("w2: ", w2)

        self.nb_z[k] += 1  # self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。
        # print("self.nb_z[k]: ", self.nb_z[k])
        self.nwz[k][w1] += 1  # self.nwz[1][1] 表示的是在主题1中，1号单词出现的次数。
        self.nwz[k][w2] += 1  # self.nwz[2][3] 表示的是在主题2中，2号单词出现的次数。
        print("在主题k中，实体1出现的总次数：   ", self.nwz[k][w1])
        print("在主题k中，实体2出现的总次数:  ", self.nwz[k][w2])

        # *****************************************************************************************
        # ***************************************************************************************

        fr_id_word = open("../output/model/voca-id-word/" + filename, 'r', encoding='utf-8')
        id = []
        word = []
        content = fr_id_word.readlines()
        # print("content: ", content)
        for line in content:
            # print(line)
            line = line.strip().split('\t')
            id.append(line[0])
            word.append(line[1])
        id_word = {id[i]: word[i] for i in range(len(id))}
        print(id_word)  # 将词典中的id和word写作字典
        words = []
        if w1 < w2:
            words.append(id_word[str(w1)])
            words.append(id_word[str(w2)])
        else:
            words.append(id_word[str(w2)])
            words.append(id_word[str(w2)])
        print("words: ", words)  # 一篇文档的词典中id和word构成的字典，便于根据id查找关系二元组中的两个实体
        print("888888888888888: ", words[0] + ',' + words[1])
        fr1 = open('../原始数据/第1份实体对/' + filename)
        fr2 = open('../原始数据/第1份实体对-1/' + filename)
        fr3 = open('../原始数据/第1份实体对-2/' + filename)
        lines1 = ""
        lines2 = ""
        lines3 = ""
        for line1 in fr1.readlines():
            lines1 = lines1 + line1.strip() + '\t\t'
        for line2 in fr2.readlines():
            lines2 = lines2 + line2.strip() + '\t\t'
        for line3 in fr3.readlines():
            lines3 = lines3 + line3.strip() + '\t\t'
        # print("lines1:  ", lines1)
        # print("lines2:  ", lines2)

        if step == 1:
            self.nw1_step1 = self.nwz[k][w1]
            self.nw2_step1 = self.nwz[k][w2]

        if step == 2:
            if (words[0]+','+words[1]) in lines2 and (words[0]+','+words[1]) not in lines1:
                self.nw1_step2 += 1  # 实体1在主题k中，且在步长为2的关系中的次数
                self.nw2_step2 += 1
            self.nw1_step1 = self.nwz[k][w1] - self.nw1_step2
            self.nw2_step1 = self.nwz[k][w2] - self.nw2_step2

        if step == 3:
            if (words[0] + ',' + words[1]) in lines3 and (words[0] + ',' + words[1]) not in lines1 and (words[0] + ',' + words[1]) not in lines2:
                self.nw1_step3 += 1  # 实体1在主题k中，且在步长为2的关系中的次数
                self.nw2_step3 += 1
            if (words[0] + ',' + words[1]) in lines2 and (words[0] + ',' + words[1]) not in lines1 and (words[0] + ',' + words[1]) not in lines3:
                self.nw1_step2 += 1  # 实体1在主题k中，且在步长为2的关系中的次数
                self.nw2_step2 += 1
            self.nw1_step1 = self.nwz[k][w1] - self.nw1_step2 - self.nw1_step3
            self.nw2_step1 = self.nwz[k][w2] - self.nw2_step2 - self.nw1_step3


        print("self.nw1_step1:  ", self.nw1_step1)
        print("self.nw1_step2:  ", self.nw1_step2)
        print("self.nw1_step3:  ", self.nw1_step3)
        print("self.nw2_step1:  ", self.nw2_step1)
        print("self.nw2_step2:  ", self.nw2_step2)
        print("self.nw2_step3:  ", self.nw2_step3)
        print("******************************************************")
        print("分母1是：", self.nw1_step1 * 1 + self.nw1_step2 * 1/2 + self.nw1_step3 * 1/3)
        print("分母2是：", self.nw2_step1 * 1 + self.nw2_step2 * 1 / 2 + self.nw2_step3 * 1 / 3)
        return self.nw1_step1 * 1 + self.nw1_step2 * 1/2 + self.nw1_step3 * 1/3, self.nw2_step1 * 1 + self.nw2_step2 * 1/2 + self.nw2_step3 * 1/3


    # 该函数的功能：计算吉布斯采样准则（论文里的公式4）某个biterm(w1,w2)主题出现的概率=该主题分布*主题词w1分布*主题词w2分布
    def comput_pz_b(self, bi, pz, bs, nw1, nw2):
        # 计算
        pz.resize(self.K)
        w1 = bi.get_wi()  # 取到词对中的第一个词编号。
        w2 = bi.get_wj()  # 取到词对中的第二个词编号。

        for k in range(self.K):
            if (self.has_background and k == 0) :
                pw1k = self.pw_b[w1]  # 单词w1的词频
                pw2k = self.pw_b[w2]
            else:
                # ================================================================================
                # 2 * self.nb_z[k] + 1
                # ================================================================================

                # pw1k = (self.nw1_step1 * 1 + self.nw1_step2 * 1/2 + self.nw1_step3 * 1/3 + self.beta) / (2 * self.nb_z[k] + self.beta + self.W * self.beta)  # 单词w1的词频
                # pw2k = (self.nw1_step2 * 1 + self.nw2_step2 * 1/2 + self.nw2_step3 * 1/3 + self.beta) / (2 * self.nb_z[k] + 1 + self.beta + self.W * self.beta)
                pw1k = (self.nw1_step1 * 1 + self.nw1_step2 * 1 / 2 + self.nw1_step3 * 1 / 3 + self.beta) / (
                        nw1 + self.beta + self.W * self.beta)  # 单词w1的词频
                pw2k = (self.nw1_step2 * 1 + self.nw2_step2 * 1 / 2 + self.nw2_step3 * 1 / 3 + self.beta) / (
                        nw2 + 1 + self.beta + self.W * self.beta)

            # ================================================================================
            # pk不需要修改
            pk = (self.nb_z[k] + self.alpha) / (len(bs) + self.K * self.alpha)  # len(self.bs)表示的是在文档中以后多少的词对
            pz[k] = pk * pw1k * pw2k


    # 该函数的功能：将得到的p(z)和p(w|z)的概率写入文件k3.pw_z和k3.pz中，主题的概率和某主题下单词的概率
    def save_res(self, res_dir):
        pt = res_dir + "pz"
        print("\nwrite p(z): "+pt)
        self.save_pz(pt)

        pt2 = res_dir + "pw_z"
        print("write p(w|z): "+pt2)
        self.save_pw_z(pt2)


    # p(z) is determinated by the overall proportions of biterms in it
    # 函数计算的是每个主题的分布。
    def save_pz(self, pt):
        pz = Pvec(pvec_v=self.nb_z)
        pz.normalize(self.alpha)
        pz.write(pt)


    # 函数计算的是每个主题下各个单词的分布
    def save_pw_z(self, pt):
        pw_z = np.ones((self.K,self.W))  # 生成K行W列的矩阵。用来保存每个主题中，各个单词出现的概率。
        wf = open(pt, 'w')
        # print("=============: ", self.nwz)
        for k in range(self.K):
            for w in range(self.W):
                # 词典中的每个单词在不同步长中出现的次数
                pw_z[k][w] = (self.nwz[k][w] + self.beta + self.beta) / (2 * self.nb_z[k] + self.W * self.beta)  # 计算每个词在这个主题中出现的概率。

                wf.write(str(pw_z[k][w]) + ' ')
            wf.write("\n")


