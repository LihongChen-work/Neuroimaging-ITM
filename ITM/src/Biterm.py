# -*- coding: utf-8 -*-


class Biterm():
    """
    @description: 词对
    @return: Biterm返回的是一个字符串“单词1   单词2     主题”

    """
    wi = 0
    wj = 0
    z = 0

    def __init__(self, w1=None, w2=None, s=None):
        if w1 is not None and w2 is not None:
            self.wi = min(w1, w2)
            self.wj = max(w1, w2)
        elif w1 is None and w2 is None and s is not None:
            w = s.split('\t')  # s 按空格分开（此处的s应该是字符串）
            self.wi = w[0]  # 字符串s的第一个词
            self.wj = w[1]
            self.z = w[2]

    def get_wi(self):
        return self.wi

    def get_wj(self):
        return self.wj

    def get_z(self):
        return self.z

    def set_z(self, k):
        self.z = k

    def reset_z(self):
        self.z = -1

    def str(self):
        _str = ""
        _str += str(self.wi) + '\t' + str(self.wj) + '\t\t' + str(self.z)
        return _str  # _str的形式是：词   词       主题

if __name__ == '__main__':
    s = "我们,开心"
    b = Biterm(s)
    print(b.str())


