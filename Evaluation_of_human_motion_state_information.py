import numpy as np
import pandas as pd
"""
Imputer将在未来被弃用使用sklearn.impute.SimpleImputer代替
from sklearn.preprocessing import Imputer #从sklearn中导入数据预处理模块Imputer
"""
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split #自动生成训练集和测试机模块
from sklearn.metrics import classification_report #预测结果评估模块

from sklearn.neighbors import KNeighborsClassifier #K近邻分类器
from sklearn.tree import DecisionTreeClassifier #决策树分类器
from sklearn.naive_bayes import GaussianNB #高斯朴素贝叶斯函数

def load_dataset(feature_paths,label_paths):
    """读取特征文件列表和标签文件列表中的内容，归并后返回"""
    feature = np.ndarray(shape=(0,41)) #定义列数量和特征维度一致为41
    label = np.ndarray(shape=(0,1)) #定义空的标签变量，0行1列

    for file in feature_paths:
        #使用逗号分隔符读取特征数据，问号代表缺失值，文件中不包含表头
        with open(file) as f:
            df = pd.read_table(f,delimiter=',',na_values='?',header=None)
        #使用平均值补全缺失值，然后将数据补全
        """
        imp = Imputer(missing_value='NaN',strategy='mean',axis=0) #axis参数表示在第几维做运算
        """
        imp = SimpleImputer(missing_values=np.nan,strategy='mean')
        imp.fit(df) #fit函数用于训练预处理器
        df = imp.transform(df) #transform函数用于生成预处理结果
        feature = np.concatenate((feature,df)) #将读入的数据合并到特征集合中
    
    for file in label_paths:
        with open(file) as f:
            df = pd.read_table(f,header=None)
            label = np.concatenate((label,df)) #将读入的数据合并到标签集合中
    label = np.ravel(label) #将标签规整为一维向量 （此处是列向量变行向量）
    return feature,label

if __name__ == '__main__':
    #设置数据路径
    """
    路径中含有中文，最好先使用f = open(filePath)函数打开，再pd.read_table(f)
    """
    feature_paths = ['D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\A.feature',
                     'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\B.feature',
                     'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\C.feature',
                     'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\D.feature',
                     'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\E.feature']
    label_paths = ['D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\A.label',
                   'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\B.label',
                   'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\C.label',
                   'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\D.label',
                   'D:\学习\Python机器学习应用-北京理工大学-礼欣、嵩天\课程数据\分类\E.label']

    #将前4个数据作为训练集读入
    x_train,y_train = load_dataset(feature_paths[:4],label_paths[:4])

    #将最后1个数据作为测试集读入
    x_test,y_test = load_dataset(feature_paths[4:],label_paths[4:])

    x_train,x_,y_train,y_ = train_test_split(x_train,y_train,test_size=0.0) #将数据随机打乱

    #创建K近邻分类器，并在测试集上进行预测
    print("Start training knn...")
    knn = KNeighborsClassifier().fit(x_train,y_train)
    print("Training done!")
    answer_knn = knn.predict(x_test)
    print("Prediction done!")

    #创建决策树分类器
    print("Start training DT...")
    dt = DecisionTreeClassifier().fit(x_train,y_train)
    print("Training done!")
    answer_dt = dt.predict(x_test)
    print("Prediction done!")

    #创建贝叶斯分类器
    print("Start training Bayes...")
    gnb = GaussianNB().fit(x_train,y_train)
    print("Training done!")
    answer_gnb = gnb.predict(x_test)
    print("Prediction done!")

    #计算准确率与召回率
    #classification_report函数从
    #精确度(precision)、召回率(recall)、f1-score、和支持度(support)四个维度进行衡量
    print("\n\nThe classfication report for knn:")
    print(classification_report(y_test,answer_knn))
    print("\n\nThe classfication report for dt:")
    print(classification_report(y_test,answer_dt))
    print("\n\nThe classfication report for gnb:")
    print(classification_report(y_test,answer_gnb))
