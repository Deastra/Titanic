import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')
import missingno as msno
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMRegressor
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt

# Variable Description

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
data=train.copy()

train.info()
Y="Survived"

train.set_index('PassengerId',inplace=True)

# Age is normal -- FIllna with mode / KNN
sns.boxplot(train["Age"]) 
plt.show()

train.nunique()

#Embarked -- fillna KNN
#Cabin -- filna A0
#Embarked -- dropna
#Name -- drop column

train_desc=train.describe().T
train_desc


sns.boxplot(train["Fare"])
plt.show()

#Fare -- outliers cleaning


#################################
# Data Analisys and Visualization
#Exploratory Data Analysis (EDA)

#Categorical Variables
cat_train=train.select_dtypes("object")
cat_cols=cat_train.columns
cat_cols

cat_train.nunique()
cat_train_group1=cat_train.groupby("Embarked").count()

sns.countplot(cat_train["Embarked"])
plt.show()

plt.figure(figsize=[14, 6])
plt.pie(x=cat_train['Embarked'].value_counts(), autopct="%.2f",labels=cat_train['Embarked'].value_counts().index)


sns.countplot(cat_train["Sex"])
plt.show()

plt.figure(figsize=[14, 6])
plt.pie(x=cat_train['Sex'].value_counts(), autopct="%.2f",labels=cat_train['Sex'].value_counts().index)

#Numerical Variables
num_train=train._get_numeric_data()
num_cols=num_train.columns
num_train.describe().T


plt.figure(figsize=[14, 6])
plt.pie(x=num_train['Survived'].value_counts(), autopct="%.2f",labels=["No","Yes"])


for col in num_cols:
    sns.boxplot(num_train[col])
    plt.show()

sns.scatterplot(x=num_train["Age"],y=num_train["Fare"],hue=num_train["Survived"])
plt.show()
#Survived ones mostly have higher fare not depending on age

sns.scatterplot(x=num_train["Age"],y=num_train["Fare"],hue=train["Sex"])
plt.show()

sns.scatterplot(x=num_train["Age"],y=num_train["Pclass"],hue=train["Survived"])
plt.show()
#Survived ones mostly from 1st class and 2nd class

sns.scatterplot(x=num_train["Survived"],y=num_train["Fare"],hue=train["Sex"])
plt.show()
#Mostly females survived

sns.scatterplot(x=num_train["Fare"],y=num_train["Survived"],hue=train["Embarked"])
plt.show()
#Seems no survivals there from QueensPort

f, ax = plt.subplots(figsize= [14, 8])
sns.heatmap(train.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

sns.pairplot(train, diag_kind='kde', markers='+')

for i in num_cols:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    sns.histplot(train[i], bins=10, ax=axes[0])
    axes[0].set_title(i)
    
    sns.boxplot(train[i], ax=axes[1])
    axes[1].set_title(i)
   
    sns.kdeplot(train[i], ax=axes[2])
    axes[2].set_title(i)
    plt.show()


##############################
#Outliers

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
        
    return dataframe

def knn_imputer(df, n):
    imputer = KNNImputer(n_neighbors = n)
    df_filled = imputer.fit_transform(df)
    df_knn_imp = pd.DataFrame(df_filled,columns = df.columns)
    return df_knn_imp

train=replace_with_thresholds(train,"Fare")
sns.boxplot(train["Fare"])


train1=train.copy()

train.info()

train["Cabin"].fillna("A0",inplace=True)
train1=train.copy()

sns.boxplot(train["Age"])
train["Age"].mean()
train["Age"].fillna(train["Age"].mean(),inplace=True)

train_desc=train.describe().T
train2=train.copy()

train.info()

train.dropna(inplace=True)

train.info()
#########################

def find_corr(df, num_col_names, limit=0.55,column="Survived"):
    high_corrs={}
    for col in num_col_names:
        if col==column:
            pass
        else:
            corr=df[[col, column]].corr().loc[col, column]
            print(col, corr)
            if abs(corr)>limit:
                high_corrs[col]=corr
    return high_corrs

find_corr(train,num_cols)
#NO high corrreleatins

train.info()
train.nunique()


#All names are unique, so droping the column 'Name'
train.drop("Name",axis=1,inplace=True)
train.info()

# MODELING

train_copy=train.copy()

train=pd.get_dummies(train,drop_first=True)
train.name="train_df"

def reg_model(df, Y, algo, test_size=0.20):
    X=df.drop(Y, axis=1)
    Y=df[[Y]]
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=test_size, random_state=42)
    model=algo.fit(X_train, Y_train)
    Y_train_pred=model.predict(X_train)
    
    # train_rmse=np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    train_acc=accuracy_score(Y_train, Y_train_pred)
    print(df.name)
    print(type(model).__name__)
    
    print("Train Score: {}".format(train_acc))
    
    Y_test_pred=model.predict(X_test)
    # test_rmse=np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    test_acc=accuracy_score(Y_test, Y_test_pred)
    print("Test Score: {}".format(test_acc))
    print('###################################')
    return (df.name, type(model).__name__, train_acc, test_acc)


# GridSearchCV

def grid_tuning(model,parameters,df,Y,test_size=0.2):
    X=df.drop(Y, axis=1)
    Y=df[[Y]]
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=test_size, random_state=42)
    model=model.fit(X_train, Y_train)
    # Y_train_pred=model.predict(X_train)
    
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    grid_obj = GridSearchCV(model, parameters,scoring='accuracy',error_score=0)
    grid_obj = grid_obj.fit(X_train, Y_train)
    model = grid_obj.best_estimator_
    
    return model

### GSCV KNN

knn = KNeighborsClassifier()

parameters_knn = {'n_neighbors': range(1,15), 
              'weights': ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [10, 20, 30, 50]
             }
model=grid_tuning(knn,parameters_knn,train,"Survived")
grid_knn=reg_model(train, "Survived", model)

### GSCV SVM

# Create a Support Vector Classifier
svc = svm.SVC()

# Hyperparameter Optimization
parameters_svc = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

model=grid_tuning(svc,parameters_svc,train,"Survived")
grid_svc=reg_model(train, "Survived", model)


### GSCV Logistic
log=LogisticRegression()


solvers =['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
penalty = ['none','l1','l2','elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]
parameters_log = dict(solver=solvers,penalty=penalty,C=c_values)

model=grid_tuning(log,parameters_log,train,"Survived")

grid_log=reg_model(train,"Survived",model)


#########################
### Grid Results to DataFrame for comparing


grids=[grid_knn,grid_svc,grid_log]

grid_results={'frame':[], 'model':[], 'train_score':[], 'test_score':[]}

for grid in grids:
    grid_results['frame'].append(grid[0])
    grid_results['model'].append(grid[1])
    grid_results['train_score'].append(grid[2])
    grid_results['test_score'].append(grid[3])

grid_results=pd.DataFrame(grid_results)


##################################################
##################################################
##################################################
# Algorithms with default parameters

results={'frame':[], 'model':[], 'train_score':[], 'test_score':[]}

knn = KNeighborsClassifier(n_neighbors=5)
gnb = GaussianNB()
svc = svm.SVC()
log=LogisticRegression()
lin_svc = svm.LinearSVC()
models=[knn,gnb,svc,log,lin_svc]

for m in models:
    res=reg_model(train,"Survived",m)
    results['frame'].append(res[0])
    results['model'].append(res[1])
    results['train_score'].append(res[2])
    results['test_score'].append(res[3])

#######################################
### Best=80.3 logistic
results_default=pd.DataFrame(results)

#######################################
# Model Tuning

##########################################  
##########################################  
########################################## 
# KNN
results_knn={'frame':[], 'model':[], 'train_score':[], 'test_score':[],"knn":[]}\


for n in range(1,15):
    print(f"**** KNN n_neighbors={n} ****")
    knn=KNeighborsClassifier(n_neighbors=n)
    res=reg_model(train,"Survived",knn)
    results_knn['frame'].append(res[0])
    results_knn['model'].append(res[1])
    results_knn['train_score'].append(res[2])
    results_knn['test_score'].append(res[3])
    results_knn["knn"].append(n)
#################################
### best=74 knn=11
results_knn=pd.DataFrame(results_knn)


##########################################  
##########################################  
##########################################  
# SVM

results_svc={'frame':[], 'model':[], 'train_score':[], 'test_score':[],"C":[],"Gamma":[]}
gamma=[0.00011,0.001,0.01,0.1,1,2,3,4,5,6,7,8,9]
C=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,500,700,1000]
for c in C:
    for g in gamma:
        print(f"**** SVC gamma={g} || C={c} ****")
        svc=svm.SVC(gamma=g,C=c)
        # reg_model(train,"Survived",svc)
        res=reg_model(train, 'Survived', svc)
        results_svc['frame'].append(res[0])
        results_svc['model'].append(res[1])
        results_svc['train_score'].append(res[2])
        results_svc['test_score'].append(res[3])
        results_svc["C"].append(c)
        results_svc["Gamma"].append(g)

#####################################
### best=82.6 C=100 Gamma=0.00011 ###
results_svc=pd.DataFrame(results_svc)

##########################################  
##########################################  
########################################## 
# Logistic
solvers =['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
penalty = ['none','l1','l2','elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]

results_log={'frame':[], 'model':[], 'train_score':[], 'test_score':[],"Solvers":[],"Penalty":[],"C":[]}
for s in solvers:
    for p in penalty:
        for c in c_values:
            try:
                log=LogisticRegression(solver=s,penalty=p,C=c)
                print(f"**** SVC Solver={s}|| penalty={p} || C={c} ****")
                res=reg_model(train, "Survived", log)
                results_log['frame'].append(res[0])
                results_log['model'].append(res[1])
                results_log['train_score'].append(res[2])
                results_log['test_score'].append(res[3])
                results_log["C"].append(c)
                results_log["Solvers"].append(s)
                results_log["Penalty"].append(p)
            except:
                pass

##############################################
### best=82.6 C=10 Penalty=l2 Solver=liblinear
results_log=pd.DataFrame(results_log)


##################
# ALL RESULTS
all_results=[
            results_default,  #Log                     0.803
            results_knn,      #KNN 11                  0.74
            results_log,      #Log liblinear/l2/10     0.826
            results_svc,      #SVC C100/Gamma0.00011   0.826
            grid_results      #Logistic                0.81
            ]



