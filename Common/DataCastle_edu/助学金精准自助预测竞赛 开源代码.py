# -*-coding:utf-8 -*-

#图书借阅数据borrow_train.txt和borrow_test.txt、(学生id，借阅日期，图书名称，图书编号)     
#一卡通数据card_train.txt和card_test.txt、(学生id，消费类别，消费地点，消费方式，消费时间，消费金额，剩余金额)  
#寝室门禁数据dorm_train.txt和dorm_test.txt、(学生id，具体时间，进出方向(0进寝室，1出寝室))
#图书馆门禁数据library_train.txt和library_test.txt、(学生id，门禁编号，具体时间)    
#学生成绩数据score_train.txt和score_test.txt、(学生id,学院编号,成绩排名)
#助学金获奖数据subsidy_train.txt和subsidy_test.txt、(学生id,助学金金额（分隔符为半角逗号）)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# train_test
train = pd.read_table('../train/subsidy_train.txt',sep=',',header=-1)
train.columns = ['id','money']
test = pd.read_table('../test/studentID_test.txt',sep=',',header=-1)
test.columns = ['id']
test['money'] = np.nan
train_test = pd.concat([train,test])

# score
score_train = pd.read_table('../train/score_train.txt',sep=',',header=-1)
score_train.columns = ['id','college','score']
score_test = pd.read_table('../test/score_test.txt',sep=',',header=-1)
score_test.columns = ['id','college','score']
score_train_test = pd.concat([score_train,score_test])

#college-统计各学院人数
college = pd.DataFrame(score_train_test.groupby(['college'])['score'].max())
college.to_csv('../input/college.csv',index=True)
college = pd.read_csv('../input/college.csv')
college.columns = ['college','num']

score_train_test = pd.merge(score_train_test, college, how='left',on='college')
score_train_test['order'] = score_train_test['score']/score_train_test['num']
train_test = pd.merge(train_test,score_train_test,how='left',on='id')

# card
card_train = pd.read_table('../train/card_train.txt',sep=',',header=-1)
card_train.columns = ['id','consume','where','how','time','amount','remainder']
card_test = pd.read_table('../test/card_test.txt',sep=',',header=-1)
card_test.columns = ['id','consume','where','how','time','amount','remainder']
card_train_test = pd.concat([card_train,card_test])

card = pd.DataFrame(card_train_test.groupby(['id'])['consume'].count())
card['consumesum'] = card_train_test.groupby(['id'])['amount'].sum()
card['consumeavg'] = card_train_test.groupby(['id'])['amount'].mean()
card['consumemax'] = card_train_test.groupby(['id'])['amount'].max()
card['remaindersum'] = card_train_test.groupby(['id'])['remainder'].sum()
card['remainderavg'] = card_train_test.groupby(['id'])['remainder'].mean()
card['remaindermax'] = card_train_test.groupby(['id'])['remainder'].max()

card.to_csv('../input/card.csv',index=True)
card = pd.read_csv('../input/card.csv')
train_test = pd.merge(train_test, card, how='left',on='id')

train = train_test[train_test['money'].notnull()]
test = train_test[train_test['money'].isnull()]

train = train.fillna(-1)
test = test.fillna(-1)
target = 'money'
IDcol = 'id'
ids = test['id'].values
predictors = [x for x in train.columns if x not in [target]]

# Oversample
Oversampling1000 = train.loc[train.money == 1000]
Oversampling1500 = train.loc[train.money == 1500]
Oversampling2000 = train.loc[train.money == 2000]
for i in range(5):
    train = train.append(Oversampling1000)
for j in range(8):
    train = train.append(Oversampling1500)
for k in range(10):
    train = train.append(Oversampling2000)


'''open model'''
#clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)#0.02301
##clf = RandomForestClassifier(n_estimators=500,random_state=2016)#0.022
#clf = clf.fit(train[predictors],train[target])
#clf.score(train[predictors],train[target])#G0.79212814645308927--R1.0
#result = clf.predict(test[predictors])
'''knn-'''
#from sklearn import neighbors  
#import sklearn
#knn = neighbors.KNeighborsClassifier(n_neighbors=3)  
##训练数据集  
#knn.fit(train[predictors],train[target])
##训练准确率
#score = knn.score(train[predictors],train[target])#0.9468192
##预测
#result = knn.predict(test[predictors])
'''tree'''
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(train[predictors],train[target])

print(clf.feature_importances_)
#准确率
score= clf.score(train[predictors],train[target])#1.0

#预测概率
answer = clf.predict_proba(train[predictors])[:,1]

result = clf.predict(test[predictors])


'''Save results'''
test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))

print('1000--'+str(len(test_result[test_result.subsidy==1000])) + ':741')
print('1500--'+str(len(test_result[test_result.subsidy==1500])) + ':465')
print('2000--'+str(len(test_result[test_result.subsidy==2000])) + ':354')

test_result.to_csv("../output/tree_result.csv",index=False)

# '''