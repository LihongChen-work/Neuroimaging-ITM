#!/usr/bin/env python
#coding=utf-8
# translate word into id in documents
#==========================================================================
# 将文档中的词用id 表示出来
# ==========================================================================
import sys

# pt中放的是要编入索引的文档，每行都是 “实体,实体” 的格式，即原始数据；
# res_pt中放的是原始单词对应的编号，不去重
def indexFile(pt, res_pt):
    w2id = {}
    print('index file: '+str(pt))
    wf = open(res_pt, 'w', encoding='utf-8')  # res_pt中放的是输出文档索引，每行都是 "id id" 的格式
    for l in open(pt, 'r', encoding='utf-8-sig'):  # strip() 方法用于移除字符串头或尾指定的字符(默认为空格或换行符)或字符序列
        # split()默认以空格或换行符分隔开
        ws = l.strip().split()  # 这里的l应该是文档中的单行字符串，ws是处理完格式后的一行数据
        print("ws: ", ws)
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)
        wids = [w2id[w] for w in ws]  
        # print>>wf,' '.join(map(str, wids))
        print(' '.join(map(str, wids)), file=wf)

    print('write file: '+str(res_pt))
    return w2id


def write_w2id(res_pt, w2id):
    print('write:'+str(res_pt))
    wf = open(res_pt, 'w', encoding='utf-8')
    for w, wid in sorted(w2id.items(), key=lambda d: d[1]):
        print('%d\t%s' % (wid, w), file=wf)


# doc_pt：输入要编入索引的文档，每行都是 “word word” 的格式
# dwid_pt：输出文档索引后，每行都是 "id id" 的格式
# voca_pt：输出词汇表文件，每行是一个 “id word” 的格式

def run_indexDocs(argv):  # argv是一个列表
    if len(argv) < 4:
        print('Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % argv[0])
        print('\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."')
        print('\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId..."')
        print('\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"')
        exit(1)
        
    doc_pt = argv[1]
    dwid_pt = argv[2]
    voca_pt = argv[3]
    w2id = indexFile(doc_pt, dwid_pt)
    print('n(w)='+str(len(w2id)))
    write_w2id(voca_pt, w2id)
    return len(w2id)
