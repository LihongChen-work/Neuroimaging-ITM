



class even:
    def __init__(self,trueNER_List,predNER_List,trueRel_List, predRel_List):
        """"初始化"""
        self.trueNER_List=trueNER_List
        self.predNER_List=predNER_List
        self.trueRel_List=trueRel_List
        self.predRel_List = predRel_List

    def getPrint(self):
        for batch_idx in range(len(self.trueNER_List)):
            print("predNER" + str(batch_idx))
            print(self.predNER_List[batch_idx])
            print("trueNER" + str(batch_idx))
            print(self.trueNER_List[batch_idx])
            print("predRel" + str(batch_idx))
            print(self.predRel_List[batch_idx])
            print("trueRel" + str(batch_idx))
            print(self.trueRel_List[batch_idx])
