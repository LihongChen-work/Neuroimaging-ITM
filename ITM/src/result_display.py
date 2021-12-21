
from document import Document



def read_voca(pt):
    voca = {}
    for l in open(pt, encoding='utf-8'):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca


def read_pz(pt):
    return [float(p) for p in open(pt).readline().split()]


def load_docs(docs_pt):
    bs = []
    print("load docs: " + docs_pt)
    rf = open(docs_pt)
    if not rf:
        print("file not found: " + docs_pt)
    for line in rf.readlines():
        d = Document(line)
        biterms = []
        d.gen_biterms(biterms)
            # statistic the empirical word distribution
        # for i in range(d.size()):
        #     w = d.get_w(i)
        #     pw_b[w] += 1
        for b in biterms:
            bs.append(b)
    # print(len(bs))
    return bs


def perplexity(argvs):
    prob_sum = 0  # p(b)
    model_dir = argvs[1]  # 模型的存储路径
    K = int(argvs[2])  # 主题个数
    voca_pt = argvs[3]  # 词汇-id对应表路径
    test_corpus = argvs[4]  # 测试集路径
    voca = read_voca(voca_pt)  # 以字典形式存储词汇id
    W = len(voca)  # 词汇个数
    pz_pt = model_dir + 'k%d.pz' % K  # 主题概率的存储路径
    pz = read_pz(pz_pt)
    zw_pt = model_dir + 'k%d.pw_z' % K  # 主题词汇概率分布的存储路径
    k = 0
    topics = []
    for l in open(zw_pt, encoding='utf-8-sig'):
        app1 = {}  # 以字典形式存储主题下词汇与其对应的概率值
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)
        wvs = sorted(wvs, key=lambda d:d[1], reverse=True)
        for w,v in wvs:
            app1[voca[w]] = v
        topics.append((pz[k], app1))  # 存储到列表：主题-词汇-概率
        # print(len(topics))
        k += 1
    # bs = cidui(test_corpus)  # 获取测试集中的词对
    bs = load_docs(test_corpus)
    for bi in bs[0:5]:
        pass



# voca-id-word = {id:w,...}
def dispTopics(pt, voca, pz, filename):
    k = 0
    topics = []
    # Path = '../BTM_result_all/'
    # Path = '../relational-result-123/'
    Path = "../实体关系结果/第1份实体对-2结果/"
    with open(Path + filename, 'w', encoding='utf-8') as fw:
        for l in open(pt):
            vs = [float(v) for v in l.split()]
            wvs = zip(range(len(vs)), vs)
            wvs = sorted(wvs, key=lambda d:d[1], reverse=True)
            #tmps = ' '.join(['%s' % voca-id-word[w] for w,v in wvs[:10]])
            tmps = ' '.join(['%s:%f#' % (voca[w], v) for w, v in wvs[:5]])  # 此处可以修改显示主题词的个数
            topics.append((pz[k], tmps))
            k += 1
        print('p(z)\t\tTop words', file=fw)
        for pz, s in sorted(topics, reverse=True):
            print('%f\t%s' % (pz, s), file=fw)


def run_topicDicplay(argv, filename):
    if len(argv) < 4:
        print('Usage: python %s <model_dir> <K> <voca_pt>' % argv[0])
        print('\tmodel_dir    the output dir of My_BTM')
        print('\tK    the number of topics')
        print('\tvoca_pt    the vocabulary file')
        exit(1)

    model_dir = argv[1]
    K = int(argv[2])
    voca_pt = argv[3]
    voca = read_voca(voca_pt)
    W = len(voca)
    print('K:%d, n(W):%d' % (K, W))

    pz_pt = model_dir + 'k%d.pz' % K
    pz = read_pz(pz_pt)  # pz是各个主题出现的概率.

    zw_pt = model_dir + 'k%d.pw_z' % K  # zw_pt是每个主题中各个单词出现的概率
    dispTopics(zw_pt, voca, pz, filename)


if __name__ == "__main__":
    mode = "est"
    K = 3
    W = None
    alpha = 0.5
    beta = 0.5
    n_iter = 10  # 十次迭代
    save_step = 100
    dir = "../output/"
    input_dir = "../sample-data/"
    model_dir = dir + "model/"  # 模型存放的文件夹
    voca_pt = dir + "voca-id-word.txt"  # 生成的词典
    dwid_pt = dir + "doc_wids.txt"  # 每篇文档由对应的序号单词组成
    doc_pt = input_dir + "trndocs.dat"  # 输入的文档
    argvs = []
    argvs.append(mode)
    argvs.append(model_dir)
    argvs.append(K)
    argvs.append(voca_pt)
    argvs.append(dwid_pt)
    # perplexity(argvs)
    # load_docs(dwid_pt)
