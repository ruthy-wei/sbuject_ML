## -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 随机抽取80%的数据作为训练集，20%的数据作为测试集
def get_train_test_data(filename,colname,label):
    data=pd.read_excel(filename)
    data_X=data[colname]
    data_Y =data[label]
    # 80%作为训练集，20%作为测试集
    # print(data_X)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2,stratify=data_Y)
    # print(y_train)
    return X_train, X_test, y_train, y_test

# 训练并测试模型-NB
def train_model_NB(X_train, X_test, y_train, y_test):
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    encoder=LabelEncoder()
    train_label=encoder.fit_transform(y_train)
    test_label=encoder.transform(y_test)

    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, train_label)
    y_predict = clf.predict(test_data)
    print(classification_report(test_label,y_predict))

# 训练并测试模型-logisticRegression
def train_model_logisticRegression(X_train, X_test, y_train, y_test):
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    encoder = LabelEncoder()
    train_label = encoder.fit_transform(y_train)
    test_label = encoder.transform(y_test)

    lr = LogisticRegression(C=1000, solver='liblinear', multi_class='auto')
    lr.fit(train_data, train_label)
    y_predict = lr.predict(test_data)
    print(classification_report(test_label, y_predict))

# 训练并测试模型-svm
def train_model_SVM(X_train, X_test, y_train, y_test=get_train_test_data):
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    encoder = LabelEncoder()
    train_label = encoder.fit_transform(y_train)
    test_label = encoder.transform(y_test)

    clf = SVC(C=1000.0,gamma='scale')
    clf.fit(train_data, train_label)
    y_predict=clf.predict(test_data)
    print(classification_report(test_label, y_predict))

# 训练并测试模型-DecisionTreeClassifier
def train_model_DecisionTreeClassifier(X_train, X_test, y_train, y_test):
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    encoder = LabelEncoder()
    train_label = encoder.fit_transform(y_train)
    test_label = encoder.transform(y_test)

    dc = DecisionTreeClassifier(max_depth=5)
    dc.fit(train_data, train_label)
    y_predict=dc.predict(test_data)
    print(classification_report(test_label, y_predict))

# 训练并测试模型-RandomForestClassifier
def train_model_RandomForestClassifier(X_train, X_test, y_train, y_test):
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    encoder = LabelEncoder()
    train_label = encoder.fit_transform(y_train)
    test_label = encoder.transform(y_test)

    rfc = RandomForestClassifier(n_estimators=1000)
    rfc.fit(train_data, train_label)
    y_predict=rfc.predict(test_data)
    print(classification_report(test_label, y_predict))

if __name__=="__main__":
    filename='F:\sbuject_ML\data\\all_merge_data_label2.xlsx'
    colname='ltp_rm_punc'
    label='sentence_subject'
    X_train, X_test, y_train, y_test=get_train_test_data(filename,colname,label)
    # print('*'*10,'NB(贝叶斯)：')
    # train_model_NB(X_train, X_test, y_train, y_test)
    # print('*' * 10, 'logisticRegression：')
    # train_model_logisticRegression(X_train, X_test, y_train, y_test)
    # print('*' * 10, 'SVM：')
    # train_model_SVM(X_train, X_test, y_train, y_test)
    print('*' * 10, 'DecisionTreeClassifier：')
    train_model_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    print('*' * 10, 'RandomForestClassifier：')
    train_model_RandomForestClassifier(X_train, X_test, y_train, y_test)