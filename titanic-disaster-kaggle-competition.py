
#kütüphaneleri yükle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df_train=pd.read_csv("/kaggle/input/titanic/train.csv")
df_test=pd.read_csv("/kaggle/input/titanic/test.csv")


df_train.head()


df_train.info()


df_train.isnull().sum()


df=pd.concat([df_train,df_test],ignore_index=True)
df


df_train.shape,df_test.shape,df.shape


numeric_columns=df.select_dtypes(include=['number'])
numeric_columns


df['Title']=df['Name'].str.extract('([A-Za-z]+)\.')


df['Title'].value_counts()


df['Title']=df['Title'].replace(['Ms','Mlle'],'Miss')
df['Title']=df['Title'].replace(['Mme','Countess','Lady','Dona'],'Mrs')
df['Title']=df['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer'],'Mr')


df['Title'].value_counts()


df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Embarked'].isnull().sum()


df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'),inplace=True)


df.isnull().sum()


df['Fare'].mean()


df['Fare'].median()


df['Fare'].fillna(df['Fare'].median(),inplace=True)


df.isnull().sum()


y=df['Survived']
X=df.drop(['Survived','PassengerId','Name', 'Ticket','Cabin'],axis=1)
X.head()


#Kategorik verileri sayısal verilere çevirir
#One Hot Encoding
X=pd.get_dummies(X,drop_first=True) #Feature Engineering özellik mühendisliği
#Pclass aslında sayısal bir veri değil önce dtype objeye çevireceksiniz sonra dummies yapacaksınız


df['Embarked'].unique()


#Train ve testi aşağıdaki gibi ayırıyoruz
X_train=X[:891] #Train veri setinde ilk 891
X_test=X[891:]
y_train=y[:891]
y_test=y[891:]


y_test


#Model Kütüphaneleri
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB
#metrikler
from sklearn.model_selection import cross_val_score
#accuaracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def model_classification(X,y):
    '''
    X: independent variable
    y: dependent variable
    return best model and its accuracy
    '''
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
    models = [GaussianNB(),BernoulliNB(),LogisticRegression(),RandomForestClassifier(), 
              GradientBoostingClassifier(), KNeighborsClassifier(n_neighbors=5)]
    results=[]
    for model in models:
        model.fit(x_train,y_train)
        model_predict=model.predict(x_test)
        print("Model: ",model)
        print("Model Accuracy: ",accuracy_score(model_predict,y_test))
        print("Model Confusion Matrix: ",confusion_matrix(model_predict,y_test),"\n")
        print("Model Classification Report: ",classification_report(model_predict,y_test))
        print("-"*50)
        results.append(accuracy_score(model_predict,y_test))
    #best model
    best_model=models[results.index(max(results))]
    print("Best Model: ",best_model)
    print("Best Model Accuracy: ",max(results))
    models=pd.DataFrame({
        'Model':['GaussianNB','BernoulliNB','LogisticRegression','RandomForestClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier'],
        'Score':results})
    print(models.sort_values(by='Score', ascending=False, ignore_index=True))
    return best_model,max(results), confusion_matrix(model_predict,y_test)


model_classification(X_train,y_train)


gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)# Testi kaggle üzeirnden yapacağımız için artık veri setini bölmeden elimizdeki tüm veri setini gönderiyoruz.
y_pred=gb.predict(X_test)
y_pred


submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': y_pred.astype(int)})
submission


submission.to_csv('submission.csv', index=False) #data frame i csv dosyası oalrak kaydediyor




