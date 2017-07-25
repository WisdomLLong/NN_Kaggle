# -*- coding: utf-8 -*-

###############################################################################
# Part I: Exploratory Data Analysis
###############################################################################
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# maching learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Learning curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import validation_curve



###############################################################################
# Part 1: Load data
###############################################################################
#------------------------------------------------------------------------------
# Step 01:load data using panda
#------------------------------------------------------------------------------
train_df = pd.read_csv('E:/Job/GitHub_Workspace/NN_Kaggle/Titanic/Datasets/train.csv')
test_df  = pd.read_csv('E:/Job/GitHub_Workspace/NN_Kaggle/Titanic/Datasets/test.csv')
combine = [train_df, test_df]

'''
PassengerId：乘客序号；
Survived：最终是否存活（1表示存活，0表示未存活）；
Pclass：舱位，1是头等舱，3是最低等；
Name：乘客姓名；
Sex：性别；
Age：年龄；
SibSp：一同上船的兄弟姐妹或配偶；
Parch：一同上船的父母或子女；
Ticket：船票信息；
Fare：乘客票价，决定了Pclass的等级；
Cabin：客舱编号，不同的编号对应不同的位置；
Embarked：上船地点，主要是S（南安普顿）、C（瑟堡）、Q（皇后镇）。
'''

#------------------------------------------------------------------------------
# Step 02: Acquire and clean data
#@@ 小总结一下就是，对各个Data columns进行内部划分（连续的实数用范围进行划分）；之后将内部分好类的Data clumns与label对应起来，看它对label的影响大小
#------------------------------------------------------------------------------
print(train_df.head(5))
print(train_df.info()) #@@ 查看数据集的特征名、个数、类型等
print(train_df.describe())
print(train_df.describe(include=['O'])) 
#@@ 注意这里的数据类型是字母O，表示的object_这样一种类型，参考下面网页
#@@ https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html

#@@ 看来这里的dataset表示的是一行数
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna('U')
    #@@ 空白的数据用字符'U'填充
    dataset['Cabin'] = dataset.Cabin.str.extract('([A-Za-z])', expand=False)
    #@@ 小括号表示必须包含（需要提取），中括号表示的类似于一个范围条件，这里表示的是大小写26个英文字母都可以
    #@@ expand=Flase表示的是extract出来的数据类型是一列，而不是一个矩阵，暂时可以简单这么理解
    
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].map(\
    {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0, 'T':0, 'U':1}).astype(int)

train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
combine = [train_df, test_df]

# survial rate distribtion as a function of Pclass
train_df[['Pclass', 'Survived']].groupby(['Pclass'],\
     as_index=False).mean().sort_values(by='Survived', ascending=False)
#@@ 通过以Pclass为组的Survived均值，并且进行排序，来判断两者之间的联系

# obtain Title from name (Mr, Mrs, Miss etc)
#@@ 获得乘客的敬称，注意观察数据会发现，小点.前面的是敬称
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dona'], 'Royalty')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major', 'Rev'], 'Officer')
    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don', 'Sir'], 'Royalty')
    #@@ 这里的loc类似于MATLAB中的选取某行某列的数据
    dataset.loc[(dataset.Sex == 'male')  &  (dataset.Title == 'Dr'), 'Title'] = 'Mr'
    dataset.loc[(dataset.Sex == 'female')  &  (dataset.Title == 'Dr'), 'Title'] = 'Mrs'
    
#: count survived rate for different titles
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(\
        by='Survived', ascending=False)
    
# Covert 'Title' to numbers (Mr->1, Miss->2...)
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royalty':5, 'Officer':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
# Remove 'Name' and 'PassengerId' in training data, and 'Name' in testing data
train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)    
combine = [train_df, test_df]

# if age<16, set 'Sex' to Child 
for dataset in combine:
    dataset.loc[(dataset.Age<16), 'Sex'] = 'Child'
    
# Covert 'Sex' to numbers (female:1, male:0, Child: 2)
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0, 'Child':2}).astype(int)

# Guess age values using median values for age across set of Pclass and gender frature combinations
#@@ 法一，直接赋值法。
#@@ 因为combine是list类型，而combine[0]是DataFrame，后者才有groupby函数。
#@@ DataFrame类型的数据调用完函数需要重新进行赋值，只有赋值才能改变其中的数值
#@@ 有个问题是这样是train和test数据分别进行fillna的。
'''
combine[0]['Age'] = combine[0].groupby(['Sex', 'Pclass'])['Age'].transform(\
                    lambda x: x.fillna(x.mean())).astype(int)
combine[1]['Age'] = combine[1].groupby(['Sex', 'Pclass'])['Age'].transform(\
                    lambda x: x.fillna(x.mean())).astype(int)
'''

#@@ 法二，循环法
for dataset in combine:
    dataset['Age'] = dataset.groupby(['Sex', 'Pclass'])['Age'].transform(\
                    lambda x: x.fillna(x.mean())).astype(int)
#@@ 以sex和PClass作为分组依据，求他们的均值，并对Age列进行补全

# create age bands and determine correlations with survived
#@@ 是为了找到划分的Age的界线
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], \
        as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Age']<=16, 'Age'] = 0
    dataset.loc[ (dataset['Age']>16) & (dataset['Age']<=32), 'Age'] = 1
    dataset.loc[ (dataset['Age']>32) & (dataset['Age']<=48), 'Age'] = 2
    dataset.loc[ (dataset['Age']>48) & (dataset['Age']<=64), 'Age'] = 3
    dataset.loc[ dataset['Age']>64, 'Age'] = 4
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]


# Create family size from 'sibsq+parch +1'
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)
    
#create another feature called IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1
    dataset.loc[(dataset['FamilySize'] >4), 'IsAlone'] = 2
            
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#drop Parch, Sibsp, and FamilySize features in favor of IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


# Create an artfical feature combinbing PClass and Age 
for datast in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass


# fill the missing values of Embarked feature with the most commmon occureance
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False)\
    .mean().sort_values(by='Survived', ascending=False)
    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)


# fill the missing values of Fare 
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)
#@@ drop掉NAN值的数据，求完中位数(mean是均值)，放到空白的位置

# Greate FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], \
        as_index=False).mean().sort_values(by='FareBand', ascending=True)

# Convert the Fare feature to ordinal values based on the FareBand 
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]



    
    
    
    
    
    
    
    
    
    
    
    
    







