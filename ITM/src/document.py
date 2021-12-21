# -*- coding: utf-8 -*-
from Biterm import *

class Document():

    ws = []

    def __init__(self, s):
        self.ws = []
        self.read_doc(s)

    # 读取数据文档
    def read_doc(self, s):
        for w in s.split(' '):  # 词w在字符串s中，则将字符w 转换成int，并存入整型数组ws[]中
            self.ws.append(int(w.strip()))


    def size(self):
        return len(self.ws)  # 计算数组ws[]的元素个数

    # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
    def get_w(self, i):
        assert(i < len(self.ws))  # 如果非常确定使用的列表中至少含有一个元素，而且你想验证这一点，并且在其非真的时候引发一个错误
        return self.ws[i]  # 当数组ws[]中元素大于i时，则返回数组的第i个元素（是个整数）

    ''' 
      Extract biterm from a document
        'win': window size for biterm extraction  窗口大小
        'bs': the output biterms  抽取出来的词对
    '''
    def gen_biterms(self, bs, win=100):  # 此时的窗口大小设置为100，意味着整个输入文档就是一个窗口
        if (len(self.ws) < 2):
            return
        # =======================================================
        # 词典中的两个单词随意组合成词对
        # for i in range(len(self.ws)-1):
        #     for j in range(i+1, min(i+win, len(self.ws))):
        #         bs.append(Biterm(self.ws[i], self.ws[j]))
        # ==========================================================
        # 不再随意组合，直接写死，输入数据每一行是以逗号隔开的两个实体
        bs.append(Biterm(self.ws[0], self.ws[1]))
        # for biterm in bs:
        #     # print('biterm:  ',type(biterm))   # biterm 的类型是“class”，调用Biterm.py文件
        #     print('wi : ' + str(biterm.get_wi()) + ' wj : ' + str(biterm.get_wj()) + ' z : ' + str(biterm.get_z()))


if __name__ == "__main__":
    s = '0 1'
    d = Doc(s)
    bs = []
    print("test")
    d.gen_biterms(bs)
    # print("bs:    ", type(bs))  # bs 的类型是"list"
    for biterm in bs:
        # print('biterm:  ',type(biterm))   # biterm 的类型是“class”，调用Biterm.py文件
        print('wi : ' + str(biterm.get_wi()) + ' wj : ' + str(biterm.get_wj()) + ' z : ' + str(biterm.get_z()))
# 打印出的结果如下所示：
#     wi: 2 wj: 3 z= 0
#     wi: 2 wj: 4
#     wi: 2 wj: 5
#     wi: 3 wj: 4
#     wi: 3 wj: 5
#     wi: 4 wj: 5
