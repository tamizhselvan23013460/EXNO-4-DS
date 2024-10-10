# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

## FEATURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

![image](https://github.com/user-attachments/assets/e8b6ab6f-d5d7-4a4c-90fc-645509266a47)

```
df.dropna()
```
![image](https://github.com/user-attachments/assets/3b341ad1-13ee-47a6-b0e6-30464bc1f6ab)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/68ad2c2c-7146-4a75-87d4-683ed7abd46e)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/15141f2e-3804-4cc3-be44-ed923c988b6f)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/8925f6d9-bd94-4531-8c11-7f5806a52a4b)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/6792822f-08b3-4c14-b46d-217c59d87b2a)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/1245b6ac-0b34-4233-a052-df3d33f47229)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/4b817c5d-e6b0-49a5-84c4-ed0db18ab713)

## FEATURE SELECTION:

```
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv("/content/income(1) (1).csv",na_values=[" ?"])
df
```
![image](https://github.com/user-attachments/assets/7320176d-3c6e-4a11-ae08-a3600c8d65ea)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/05892438-b109-4577-a94a-fd3a3b9dd5b0)

```
missing=df[df.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/c580054f-00c1-418e-b8d7-e0908e54ed06)

```
df2=df.dropna(axis=0)
df2
```
![image](https://github.com/user-attachments/assets/c8633039-6e66-45da-a94c-5a26ad248e31)

```
sal=df['SalStat']
df2['SalStat']=df2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])
df2
```

![image](https://github.com/user-attachments/assets/57dc5843-48d2-4a28-8052-c5a168bec6ee)

```
new_data=pd.get_dummies(df2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/f960e418-02c5-48d8-b0cb-ae7a42c7ff2d)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/fcba6639-6266-4e55-ae4e-40850d036bc1)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/2930819f-a6a8-4ef4-943e-f8c8a7c20ffa)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/1d1afb6f-03d1-450d-a707-cbc0967118f8)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/c4b2ff88-e691-4c4f-bdf9-6de3fa9736de)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/ba08df77-519b-4bcb-8ac1-1c54d1673cbe)

```
prediction=KNN_classifier.predict(test_x)
print(prediction)
```
![image](https://github.com/user-attachments/assets/e0e98fb0-624c-4033-a4f2-e9a82c1061e7)

```
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/35f201d9-ddf2-4539-b0ff-b6ca4de1ecd3)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/53d3d365-05d7-4a5b-b962-a2a845857dd4)

```
print('Misclassified samples: %d' %(test_y!=prediction).sum())
```
![image](https://github.com/user-attachments/assets/8911cdae-a822-4729-bb84-7a1e8c0a2373)

```
df.shape
```
![image](https://github.com/user-attachments/assets/13b713cd-2df9-403b-9d9f-93ff65e993ea)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/user-attachments/assets/bc2d0ea2-876d-43d1-b45c-8325685b0b36)

```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/user-attachments/assets/e870e54a-721e-4c71-a97b-bd67447305e1)

```
chi2, p, _, _=chi2_contingency(contigency_table)
print(f"Chi-square statistics:{chi2}")
print(f"p-value:{p}")
```
![image](https://github.com/user-attachments/assets/a33ae3b0-40fe-4abf-b6cc-831649f142cf)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/60f4cda9-6921-426e-8c00-98e274edd510)

```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/de067558-b9ac-4c6a-b3b3-f7bf34ba422c)


# RESULT:
Hence,Feature Scaling and Feature Selection process has been performed on the given data set.
