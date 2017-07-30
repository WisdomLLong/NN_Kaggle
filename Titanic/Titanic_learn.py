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

print(train_df.describe())
#@@ 显示train_df各个列或者说特征的均值等基本信息的统计

#correlation matrix
f, ax = plt.subplots(figsize= (12, 9))
sns.heatmap(train_df.corr(), vmax=.8, square=True)
#@@ 画出特这值之间的相关性



###############################################################################
# Part II: Learning Model
###############################################################################
    
#------------------------------------------------------------------------------
# Step 03: Learning model
#------------------------------------------------------------------------------    

# grid search
def grid_search_model(X, Y, model, parameters, cv):
    CV_model = GridSearchCV(estimator= model, param_grid=parameters, cv=cv)
    CV_model.fit(X, Y)
    CV_model.cv_results_
    print('Best Score:', CV_model.best_score_, '/ Best parameters:',\
           CV_model.best_params_)
    
# validation curve
def validation_curve_model(X, Y, model, param_name, parameters, cv, ylim, log=True):
    #@@ 1、掉函数，检索超参数
    train_scores, test_scores = validation_curve(model, X, Y, \
    param_name=param_name, param_range=parameters, cv=cv, scoring='accuracy' )
    #@@ 2、求出均值与标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #@@ 3、画图
    plt.figure()
    plt.title("Validation curve")
    plt.fill_between(parameters, train_scores_mean - train_scores_std,\
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(parameters, test_scores_mean - test_scores_std,\
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    if log==True:
        #@@ semilogx是在对x轴取对数
        plt.semilogx(parameters, train_scores_mean, 'o-', color='r', label='Training score')
        plt.semilogx(parameters, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    else:
        plt.plot(parameters, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(parameters, test_scores_mean, '0-', color='g', label='Cross-validation score')
        
    #plt.ylim([0.55, 0.9])
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.ylabel('Score')
    plt.xlabel('Parameter C')
    plt.legend(loc='best')
    return plt
    
# Learning curve
def Learning_curve_model(X, Y, model, cv, train_sizes):
    plt.figure()
    plt.title('Learning curve')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
        
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, \
            cv=cv, n_jobs=4, train_sizes=train_sizes)
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
        
    plt.fill_between(train_scores_mean - train_scores_std,\
                     train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(test_scores_mean - test_scores_std,\
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(test_scores_mean, '0-', color='g', label='Cross-validation score')
    #@@ legend放到最佳的位置
    plt.legend(loc='best')
    return plt
        
# Learning, predictiong and printing results
def predict_model(X, Y, model, Xtest, submit_name):
    model.fit(X, Y)
    Y_pred = model.predict(Xtest)
    score = cross_val_score(model, X, Y, cv=cv)
    
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], \
                               'Survived': Y_pred})
    #@@ 保存为CSV数据
    submission.to_csv(submit_name, index=False)
    return score

###############################################################################
# data and cv
X_data = train_df.drop('Survived', axis=1)  #data:Features
Y_data = train_df['Survived']   #data: Labels
X_test_kaggle = test_df.drop('PassengerId', axis=1).copy()  #test data(Kaggle)
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#@@ cv是cross validation的缩写；ShuffleSpli函数在这里是将1:100的数据打乱顺序，80%作为训练数据，20%作为测试数据


###############################################################################
# Logistic Regression
search_param = 0    # 1--grid search / 0--don't search
plot_vc      = 0    # 1--display validation curve / 0--don't display
plot_lc      = 1    # 1--display learning curve / 0--don't display

# grid search: logistic Regression
model = LogisticRegression()
if search_param==1:
    #@@ logspace(a,b,N) 把10的a次方到10的b次方区间分成N份
    param_range = np.logspace(-6, 5, 12)
    param_grid  = dict(C=param_range)
    grid_search_model(X_data, Y_data, model, param_grid, cv)
    
#Validation Curve: Logistic Regression
if plot_vc==1:
    param_range = np.logspace(-6, 3, 10)
    param_name = 'C'
    ylim=[0.55, 0.9]
    validation_curve_model(X_data, Y_data, model, 'C', param_range, cv, ylim)
    
#learn curve
logreg = LogisticRegression(C=1000)   
if plot_lc ==1:
    train_size = np.linspace(.1, 1.0, 5)
    #@@ 从0.1到1.0的15个等距离数，因为learning curve比较占内存，因此可以把数改小一些
    Learning_curve_model(X_data, Y_data, logreg, cv, train_size)
    
# Logistic Regression
acc_log = predict_model(X_data, Y_data, logreg, X_test_kaggle, 'submission_Logistic.csv')


###############################################################################
# Support Vector Machines
search_param = 0
plot_vc      = 0
plot_lc      = 0

# grid searhc: SVM
if search_param == 1:
    param_range = np.linspace(0.5, 5, 9)
    param_grid = dict(C=param_range)
    grid_search_model(X_data, Y_data, SVC(), param_grid, cv)
    
# Validation Cureve: SVC
if plot_vc == 1:
    param_range = np.linspace(0.1, 10, 10)
    param_name = 'C'
    ylim = [0.78, 0.90]
    validation_curve_model(X_data, Y_data, SVC(), 'C', param_range, cv, ylim, log=False)
    
# learn curve: SVC
svc = SVC(C=1, probability=True)
if plot_lc ==1:
    train_size = np.linspace(.1, 1.0, 5)
    Learning_curve_model(X_data, Y_data, svc, cv, train_size)
    
# Support Vector Machines
acc_svc = predict_model(X_data, Y_data, svc, X_test_kaggle, 'submission_SVM.csv')

    
###############################################################################
# KNN
search_param = 0
plot_vc      = 0
plot_lc      = 0

# grid search: KNN
if search_param == 1:
    param_range = (np.linspace(1, 10, 10)).astype(int)
    param_grid = dict(n_neighbors = param_range)
    grid_search_model(X_data, Y_data, KNeighborsClassifier(), param_grid, cv)
    
# Vlidation Curve: KNN
if plot_vc == 1:
    param_range = np.linspace(2, 20, 10).astype(int)
    param_name = 'n_neighbors'
    ylim = [0.75, 0.90]
    validation_curve(X_data, Y_data, KNeighborsClassifier(),\
                     'n_neighbors', param_range, cv, ylim, log = False)
    
# learn curve: KNN
knn = KNeighborsClassifier(n_neighbors = 10)
if plot_lc == 1:
    train_size = np.linspace(.1, 1.0, 5)
    Learning_curve_model(X_data, Y_data, knn, cv, train_size)

# KNN
acc_knn = predict_model(X_data, Y_data, knn, X_test_kaggle, 'submission_KNN.csv')


###############################################################################
# Naive Bayes
# Gaussian Naive Bayes
gaussian = GaussianNB()
acc_gaussian = predict_model(X_data, Y_data, gaussian, X_test_kaggle,\
                    'bmission_Gassian_Naive_Bayes.csv')


###############################################################################
# Perceptron
perceptron = Perceptron()
acc_perceptron = predict_model(X_data, Y_data, perceptron, X_test_kaggle,\
                    'submission_Perception.csv')


###############################################################################
# Linear SVC
linear_svc = LinearSVC()
acc_linear_svc = predict_model(X_data, Y_data, linear_svc, X_test_kaggle,\
                    'submission_Linear_SVC.csv')


###############################################################################
# Stochastic Gradient Descent
sgd = SGDClassifier()
acc_sgd = predict_model(X_data, Y_data, sgd, X_test_kaggle,'submission_sgd.csv')


###############################################################################
# Decision Tree
decision_tree = DecisionTreeClassifier()
acc_decision_tree = predict_model(X_data, Y_data, decision_tree, X_test_kaggle,\
                     'submission_Decision_Tree.csv')


###############################################################################
# Random Forest
search_param = 0
plot_vc      = 0
plot_lc      = 0
#@@ 第一个if是在找参数min_samples_leaf的最优解
if plot_vc == 1:
    param_range = np.linspace(10, 110, 10).astype(int)
    ylim = [0.75, 0.90]
    validation_curve_model(X_data, Y_data, RandomForestClassifier(min_samples_leaf=12),\
                'n_esti')
if plot_vc == 1:
    param_range = np.linspace(1, 21, 10).astype(int)
    ylim = [0.75, 0.90]
    validation_curve_model(X_data, Y_data, RandomForestClassifier(n_estimators=80),\
                'min_samples_leaf', param_range, cv, ylim, log = False)
    
#Random Forest
random_forest = RandomForestClassifier(n_estimators=80, random_state=0, min_samples_leaf=12)
acc_random_forest = predict_model(X_data, Y_data, random_forest, X_test_kaggle, 'submission_random_forest.csv')



###############################################################################
# Ensemble voting
#ensemble votring
ensemble_voting = VotingClassifier(estimators=[('lg', logreg), ('sv', svc), ('rf', random_forest),('kn',knn)], voting='soft')
acc_ensemble_voting = predict_model(X_data, Y_data, ensemble_voting, X_test_kaggle, 'submission_ensemble_voting.csv')

models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                                'Random Forest', 'Naive Bayes', 'Perceptron',
                                'Stochastic Gradient Decent', 'Linear SVC',
                                'Decision Tree', 'ensemble_voting'],'KFoldScore': [acc_svc.mean(), acc_knn.mean(), acc_log.mean(),
                                acc_random_forest.mean(), acc_gaussian.mean(), acc_perceptron.mean(),
                                acc_sgd.mean(), acc_linear_svc.mean(), acc_decision_tree.mean(), acc_ensemble_voting.mean()],
                                'Std': [acc_svc.std(), acc_knn.std(), acc_log.std(),
                                acc_random_forest.std(), acc_gaussian.std(), acc_perceptron.std(),
                                acc_sgd.std(), acc_linear_svc.std(), acc_decision_tree.std(), acc_ensemble_voting.std()]})

models.sort_values(by='KFoldScore', ascending=False)
    
    
    
    
    
    







