# Titanic Learn（Kaggle排名靠前解答）:
</br>1、Load data
</br>pd.read_csv('D/database/train.csv')

</br>2、Acquire and clean data
</br>目标：将String类型数据转化为int类型数据；
对int类型数据进行划分和压缩变为很小的int数据
</br>&emsp;1）head(),info(),describe()观察原始数据，好习惯
</br>&emsp;2）对每一列特征值进行fillna，正则，map，LabelEncoder
</br>&emsp;3）学会观察特征值之间的联系，进而创造出新的特征值
（就像Titanic根据兄弟姐妹和父母子女创造出了IsAlone特征值）
</br>&emsp;4）最后通过seaborn.heatmap()函数观察特征值之间的相关性

</br>3、Learning Model
</br>目标：训练和分析
</br>&emsp;1）划分好训练数据的X和Y，还有训练数据
</br>&emsp;2）定义四个函数：
</br>&emsp;&emsp;#Grid search：网格式搜索超参数
</br>&emsp;&emsp;#Validation curve：观察不同超参数对于model性能的影响，并画出图
</br>&emsp;&emsp;#Learning curve：观察不同数据量对于model性能的影响（也可看出是否欠/过拟合），并画出图
</br>&emsp;&emsp;#Learning, prediction and printing results：也就是直接进行训练和预测
</br>&emsp;3）不同的训练model创建好实例后就可以调用这些函数训练和预测了




