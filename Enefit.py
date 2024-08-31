#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
import joblib
import csv


# In[2]:


directory_path=(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data")
csv_dict=os.listdir(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data")
csv_files=[files for files in csv_dict if files.endswith(".csv")]
file_dict={}
for data_file in csv_files:
    file_path=os.path.join(directory_path,data_file)
    file_name=pd.read_csv(file_path)
    null=file_name.isnull().sum().sum()
    categorical=file_name.select_dtypes(include=['object']).columns
    file_dict[data_file]=file_name
    for i in file_name.columns:
        if i in categorical and null>0:
            replace_value=file_name[i].mode()[0]
            file_name[i].fillna(replace_value,inplace=True)
            # print(file_name[i])
        
           
        elif i not in categorical and null>0:
            replace_value=file_name[i].mean()
            file_name[i].fillna(replace_value,inplace=True)
            #print(file_name[i])
            
for name,value in file_dict.items():
    null_type=value.isnull().sum().sum()
    print(name)
    print(null_type)
    
    


# In[3]:


outlier_dict={}
for data_file in csv_files:
    file_path=os.path.join(directory_path,data_file)
    file_name=pd.read_csv(file_path)
    for columns in file_name.columns:
        if pd.api.types.is_numeric_dtype(file_name[columns]):
            q1=file_name[columns].quantile(0.25)
            q2=file_name[columns].quantile(0.75)
            iqr=q2-q1        
            lower=q1-1.5*iqr
            upper=q2+1.5*iqr
            outliers=(file_name[columns]< lower)|(file_name[columns] > upper)
            outlier_dict[columns]=outliers
            print(f"Before:")
            print(file_name[columns].describe())
outlier_list={}
for name,value in outlier_dict.items():
    if value.sum()>70:
        outlier_list[name]=value.sum()
outlier_list

    


# In[4]:


for data_file in csv_files:
    file_path=os.path.join(directory_path,data_file)
    file_name=pd.read_csv(file_path)
    for columns in file_name.columns:
        if columns in outlier_list:
            win_data=winsorize(file_name[columns],limits=[0.05,0.05])
            file_name[columns]=win_data
        print(f"After:")
        print(file_name[columns].describe())
merged_df=pd.read_csv(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data\train.csv")
new_data_files=[i for i in csv_files if i!="train.csv"]


# In[6]:


for data_file in new_data_files:
    file_path=os.path.join(directory_path,data_file)
    file_name=pd.read_csv(file_path)
    merged_df=pd.concat([merged_df,file_name],axis=0,ignore_index=True)


# In[7]:


null=merged_df.isnull()
null=null.sum()
categorical=merged_df.select_dtypes(include=['object']).columns
print(null)
print(categorical)


# In[8]:


for columns in merged_df.columns:
    if columns not in categorical:
        win_data=merged_df[columns].mean()
        merged_df[columns].fillna(win_data,inplace=True)
for columns in categorical:
    merged_df=merged_df.drop([columns],axis=1)
new_null=merged_df.isnull()
new_null=new_null.sum()
new_null


# In[10]:


q1=merged_df.quantile(0.25)
q2=merged_df.quantile(0.75)
iqr=q2-q1 
def identify(columns):
    lower=q1[columns]-1.5*iqr[columns]
    upper=q2[columns]+1.5*iqr[columns]
    outliers=(merged_df[columns]< lower)|(merged_df[columns] > upper)
    return outliers
outlier_dict={i:identify(i) for i in merged_df.columns}
outlier_list={}
for name,value in outlier_dict.items():
     if value.sum()>100:
        outlier_list[name]=value.sum()
outlier_list
for columns in merged_df.columns:
    if columns in outlier_list:
        win_data=winsorize(merged_df[columns],limits=[0.05,0.05])
        merged_df[columns]=win_data
                        


# In[10]:


attributes=[columns for columns in merged_df if columns!="target" and columns!="data_block_id" and columns!="row_id" and columns!="prediction_unit_id"]
X=merged_df[attributes].values
y=merged_df['target'].values
f_2,z=f_regression(X,y,center=True,force_finite=True)
results=pd.DataFrame({'Attributes':attributes,'P-value':f_2,'z-value':z})
results


# In[11]:


feature_dict={}
dispersion_dict={}
for columns in merged_df.columns:
    if columns in attributes:
        arr=np.array(merged_df[columns])
        mean_value=np.mean(arr)
        mad_value = np.mean(np.abs(arr-mean_value))
        feature_dict[columns]=mad_value
feature_dict


# In[12]:


correlation=np.corrcoef(X.T,y)[:,-1]
for i, coef in zip(attributes,correlation):
    print(f"Correlation coefficient between feature {i} and target: {coef}")


# In[13]:


for name,value in enumerate(attributes):
    feature_value=X[:,name]
    if np.std (feature_value)== 0:
        print(f"Feature name: {name}, Correlation: {correlation}, P-value: {p_value}")
    else:    
        correlation,p_value=pearsonr(feature_value,y)
        print(f"Feature name: {name}, Correlation: {correlation}, P-value: {p_value}")


# In[14]:


mutual_scores=mutual_info_regression(X,y)
for name,score in zip(attributes,mutual_scores):
    print(f"mutual-dependance between feature {name} and target: {score}")


# In[15]:


selected_attributes=[name for name,value in zip(attributes,mutual_scores) if value>0.5]
model_X=merged_df[selected_attributes].values
X_train,X_test,y_train,y_test=train_test_split(model_X,y,test_size=0.2,random_state=42)
Model=RandomForestRegressor()


# In[16]:


Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mae


# In[30]:


mse= mean_squared_error(y_test,y_pred)
mse


# In[87]:


Model_xgb=xgb.XGBRegressor(objective='reg:squarederror',learning_rate=0.01,max_depth=8,colsample_bytree=0.8,subsample=0.8,reg_alpha=1,n_estimators=2000)
Model_xgb.fit(X_train,y_train)
y_pred_xgb=Model_xgb.predict(X_test)


# In[88]:


mae_xgb=mean_absolute_error(y_test,y_pred_xgb)
mae_xgb


# In[73]:


directory_test=(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data\example_test_files")
csv_dir=os.listdir(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data\example_test_files")
csv_test=[files for files in csv_dir if files.endswith(".csv") and files!="sample_submission.csv" and files!="revealed_targets.csv"]
file_dict={}
for data_test in csv_test:
    file_test=os.path.join(directory_test,data_test)
    file_name=pd.read_csv(file_test)
    categorical_test=file_name.select_dtypes(include=['object']).columns
    file_dict[data_test]=file_name
    for columns in file_name.columns:
        null_test=file_name[columns].isnull().sum()
        if columns in categorical_test and null_test>0:
            replace_value=file_name[columns].mode()[0]
            file_name[columns].fillna(replace_value,inplace=True)
        elif columns not in categorical_test and null_test>0:
            replace_value=file_name[columns].mean()
            file_name[columns].fillna(replace_value,inplace=True)
    new_null=file_name.isnull()
    new_null=new_null.sum()
    print(categorical_test)   


# In[74]:


csv_test


# In[75]:


outlier_dict={}    
for data_test in csv_test:
    file_test=os.path.join(directory_test,data_test)
    file_name=pd.read_csv(file_test)
    boolean_column=file_name.select_dtypes(include=['bool']).columns
    file_name[boolean_column]=file_name[boolean_column].astype('int64')
    for columns in file_name.columns:
        if pd.api.types.is_numeric_dtype(file_name[columns]):
            q1=file_name[columns].quantile(0.25)
            q2=file_name[columns].quantile(0.75)
            iqr=q2-q1
            lower=q1-1.5*iqr
            upper=q2+1.5*iqr
            outliers=(file_name[columns]<lower)|(file_name[columns]>upper)
            outlier_dict[columns]=outliers
            print("Before winsorization:")
            print(file_name[columns].describe())    
        
outlier_list_test={}
for name ,value in outlier_dict.items():
    if value.sum()>10:
        outlier_list_test[name]=value.sum()
          


# In[76]:


outlier_list_test


# In[77]:


for data_test in csv_test:
    file_test=os.path.join(directory_test,data_test)
    file_name=pd.read_csv(file_test)
    for columns in file_name:
        if columns in outlier_list_test:
            print(f"{columns}")
            win_data=winsorize(file_name[columns],limits=[0.05,0.05])
            file_name[columns]=win_data
        print("\nAfter winsorization:")
        print(file_name[columns].describe())


# In[78]:


test=pd.read_csv(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data\example_test_files\test.csv")
new_test_files=[i for i in csv_test if i!="test.csv"]
for data_test in new_test_files:
    file_test=os.path.join(directory_test,data_test)
    file_name=pd.read_csv(file_test)
    test=pd.concat([test,file_name],axis=0,ignore_index=True)
null=test.isnull()
null=null.sum()
categorical_test=test.select_dtypes(include=['object']).columns
print(null)
print(categorical_test)


# In[82]:


for columns in test.columns:
    if columns not in categorical_test:
        win_data=test[columns].mean()
        test[columns].fillna(win_data,inplace=True)
#for columns in categorical_test:
    #print(f"{columns}")
    #test_new=test.drop([columns],axis=1)
test_new = test.drop(categorical_test, axis=1)
new_null_test=test_new.isnull()
new_null_test=new_null_test.sum()
new_null_test


# In[84]:


q1=test_new.quantile(0.25)
q2=test_new.quantile(0.75)
iqr=q2-q1 
def identify(columns):
    lower=q1[columns]-1.5*iqr[columns]
    upper=q2[columns]+1.5*iqr[columns]
    outliers=(test_new[columns]< lower)|(test_new[columns] > upper)
    return outliers
outlier_dict={i:identify(i) for i in test_new.columns}
outlier_list={}
for name,value in outlier_dict.items():
     if value.sum()>100:
        outlier_list[name]=value.sum()
outlier_list
for columns in test_new.columns:
    print(f"Before:")
    print(test_new[columns].describe())
    if columns in outlier_list:
        win_data=winsorize(test[columns],limits=[0.05,0.05])
        test_new[columns]=win_data
    print(f"After:")
    print(test[columns].describe())                       


# In[89]:


attributes_test=[columns for columns in test_new if columns!="target" and columns!="data_block_id" and columns!="row_id" and columns!="prediction_unit_id"]
selected_attributes_test=[name for name,value in zip(attributes_test,mutual_scores) if value>0.5]
model_test_x=test_new[selected_attributes_test].values
y_pred_test=Model_xgb.predict(model_test_x)


# In[90]:


y_pred_test


# In[86]:


y_test_rand=Model.predict(model_test_x)
y_test_rand


# In[72]:


row_id=test['row_id'].tolist()
data_block_id=test['data_block_id'].tolist()
row_id=np.array(row_id)
data_block_id=np.array(data_block_id)
target=y_test_rand
file_n=(r"C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Data\Sample_submission.csv")
with open(file_n,'w',newline='') as file:
    csv_w=csv.writer(file)
    csv_w.writerow(['row_id','data_block_id','target'])
    for row_id_val, data_block_id_val, target_val in zip(row_id,data_block_id,target):
        csv_w.writerow([row_id_val, data_block_id_val, target_val ])

joblib.dump(Model_xgb,r'C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Enefit\myapp\xgb_model.pkl')
joblib.dump(Model,r'C:\Users\ArtisusXiren\Desktop\predict-energy-behavior-of-prosumers\Enefit\myapp\random.pkl')
