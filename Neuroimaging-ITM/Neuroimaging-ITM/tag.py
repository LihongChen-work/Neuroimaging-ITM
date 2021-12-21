import numpy as np


fw1 = open("第1份结果-句子.txt", 'w', encoding = 'utf-8')
fw2 = open("第1份结果-实体及关系.txt", 'w', encoding = 'utf-8')


def ttag(lab_chunks, trueRel, n, i, strstr):
    fw2.write("------第"+ str(i) + "组" + strstr + "结果------" + '\n')
    fw2.write(str(lab_chunks))
    fw2.write('\n')
    fw2.write(str(trueRel))
    fw2.write('\n')
    

def old_data(lists):
   
    lists = lists.tolist() 
    stens = []
    for j in range(len(lists)):  
       
        if '#doc' in lists[j][0]:
            lists[j].insert(0, '-1')
        stens.append(lists[j][1])
   
    
    st = ""
    for i in range(len(stens)):  # range(1, len(sten))
        
        if i == 0:  
            continue
        if type(stens[i-1]) == float:
            st = st + str(stens[i-1]) + ' '
        else:
            st = st + stens[i-1] + ' '
        if stens[i-1] == '.' and '#doc' in stens[i]:
            fw1.write(st.strip() + '\n')
            # fw1.write()
            st = ""
    fw1.write(st.strip() + ' ' + stens[-1])
    








