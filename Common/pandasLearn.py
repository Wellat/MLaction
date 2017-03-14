# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:01:19 2017

@author: Vanguard
"""
from pandas import Series
import pandas as pd
import numpy as np

'''Series'''
obj=Series([4,1,2,8],index=['a','d','s','f'])

obj.values
obj.index

sdata={'ohio':3500,'ttio':2000.8,'utah':1500,'rrsd':500}

obj2=Series(sdata)

obj2[0]
obj2['ohio']
obj2[['ttio','utah']]


'''DataFrame'''
#dict
data={'shuzi':['1','2','3','9','8'],'zimu':['a','s','d','f','g']}
#getDict
data['shuzi'][0]

frame=pd.DataFrame(data)
frame=pd.DataFrame(data,columns=['zimu','shuzi'],index=['1','2','3','4','5'])

#get--column
frame.zimu
frame['zimu']
#get--row
frame.ix['1']
frame.ix[1]

frame['shuzi']=6
frame['shuzi']=np.arange(5)

cha=Series([4,1,2,8,3],index=['a','d','s','f','c'])
frame['shuzi']=cha



'''
loc:通过行标签索引
iloc:通过行号索引
ix:通过行号或行标签索引
'''
import pandas as pd  
tdata = [[1,2,3],[4,5,6]] 
index = ['d','e']  
columns=['a','b','c']  
df = pd.DataFrame(data=tdata, index=index, columns=columns)  
print(df.loc['d',['b','c']])
print(df.iloc[1,1:3])

print(df.ix['d',['b','c']])
















