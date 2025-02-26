# import numpy
import pandas as pd

class StatsCollector(object):

    def __init__(self, keys: list):
        self.collector = dict()
        for k in keys:
            self.collector[k]=list()

    def addNewObservation(self,k,v):
        self.collector.get(k).append(v)

    def setNewObservation(self,k,v):
        self.collector[k]=v

    def statsReset(self):
        self.collector.clear()

    def statsDisplay(self):
        print(self.collector)

    def saveMatrixStatsToCsv(self,k,filepath):
        pdf = pd.DataFrame(self.collector.get(k))
        pdf.to_csv(filepath,header=False,index=False)







