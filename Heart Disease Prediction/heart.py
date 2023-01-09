import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
from sklearn.svm import SVC


st.title('Heart Disease Prediction')
age = st.number_input('Enter the Age:', 1,120)
sex = st.number_input('Enter the Gender\n(For female Enter 0 and for male Enter 1):', 0,1)
chestPtype = st.number_input('Enter the ChestPainType:\n Enter 0 for ASY: \n Enter 1 for ATA:\n Enter 2 for NAP: \n Enter 3 for TA:', 0,3)
restingBp = st.number_input('Enter RestingBP:',100, 200)
chol = st.number_input('Enter the Cholestrol value:', 0, 1000)
FasBs = st.number_input('Enter the Fasting Blood Sugar:')
restingecg = st.number_input('Enter the Resting Electrocardiogram (ECG): \n Type 0 for LVH \n Type 1 for Normal \n Type 2 for ST:', 0, 2)
maxhr = st.number_input('Enter the MaxHR:')
exerAngina = st.number_input('Enter the ExerciseAngina: \n Type 1 if Yes \n Type 0 if No', 0,1)
oldpeak = st.number_input('Enter the value of Oldpeak: (Type the value in decimal)')
st_slope = st.number_input('Enter the value of ST_Slope: \n Type 0 if down: \n Type 1 if flat \n Type 2 if up', 0,2)


dataset=pd.read_csv('heart.csv')


# In[3]:


dataset.head(6)


# In[4]:


# dataset.columns


# X = 'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
#        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope  
# Y= HeartDisease

# In[5]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values  #chosing y as last column with all rows


# In[6]:


print(X)


# In[7]:


print(Y)


# ##### Data Preprocessig phase

# In[8]:


from sklearn.impute import SimpleImputer


# In[9]:


from sklearn.impute import KNNImputer


# In[10]:


from sklearn.preprocessing import LabelEncoder
#defining object for each column seperately
le1= LabelEncoder()  #denote first index
le2= LabelEncoder()  
le6= LabelEncoder()  # denotes resting ecg and goes on
le8= LabelEncoder()
le10= LabelEncoder()


# fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. 

# In[11]:


#we are transforming this into the endcoded form

X[:,1] = le1.fit_transform(X[:,1])
X[:,2] = le2.fit_transform(X[:,2])
X[:,6] = le6.fit_transform(X[:,6])
X[:,8] = le8.fit_transform(X[:,8])
X[:,10] = le10.fit_transform(X[:,10])


# In[12]:


print(X)


# see, it is encoded in the number form

# In[13]:


## Splitting Data set into training and testing dataset
## 80 percent for training
## 20 percent for testing


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test  =  train_test_split(X,Y,test_size=0.2,random_state=0)


# In[15]:


print(X_train)


# In[16]:


print(X_test)


# In[17]:


print(Y_train)


# In[18]:


print(Y_test)


# In[19]:


## see Y_test is just 20% of Y_train


# #####  Feature Scaling
# we use feature scaling , see in the data there are some values like 70 and there are also some values 0,1,2 like...
# so there is large variation in values
# so in plotting the graph its very difficulty in plotting graph,
# so if we pass into machine learning algorithm, then it will take lot of time and computation...
# and also will not give good result..
# so we perform feature scaling and transform dataset into definite range..
# 

# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


sc= StandardScaler()  ##instance of this class


# In[22]:


## X values are having large variations 
## Y dont need bcz it already has values in the form of 0 and 1


# In[23]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[24]:


print(X_train)


# In[25]:


print(X_test)


# we have completed preprocessing part upto here

# ##### Training Dataset

# ##### Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


model_logistic = LogisticRegression()  ## instance of the class

model_logistic.fit(X_train,Y_train)   ## we have to apply here curve fitting

#X_train we have written above bcz X is an independent variable
#Y_Train and Y is deep end variable


# In[ ]:





# 

# ##### SVM

# In[28]:


## SVM

from sklearn.svm import SVC
model_svm = SVC()  #instance
model_svm.fit(X_train,Y_train)


# In[29]:


y_pred_logistic = model_logistic.predict(X_test)  ## bcz we are testing model on X_test


# X_test are the parameters which are deciding Y_test

# ##### doing same with svm model ..
# we are doing this just to ensure which of our model will give better accuracy 

# In[30]:


y_pred_svm = model_svm.predict(X_test)


#   

#   

# ### KNeighbors

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
model_kneighbors = KNeighborsClassifier(n_neighbors=8)


# In[32]:


model_kneighbors.fit(X_train,Y_train)


# In[33]:


y_pred_kneighbors = model_kneighbors.predict(X_test)


#    

# ### Decision Tree

# In[34]:


from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()


# In[35]:


model_tree.fit(X_train,Y_train)


# In[36]:


y_pred_tree = model_tree.predict(X_test)


#     

#   
#   

# ###  Random  Forest 

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


model_random = RandomForestClassifier()


# In[39]:


model_random.fit(X_train,Y_train)


# In[40]:


y_pred_random  = model_random.predict(X_test)


#   

#    

# ### XGBoost

# In[41]:


from xgboost import XGBClassifier


# In[42]:


model_xg = XGBClassifier()


# In[43]:


model_xg.fit(X_train,Y_train)


# In[44]:


y_pred_xg = model_xg.predict(X_test)


#   
#     

#   

# #### testing accuracy

# In[45]:


from sklearn.metrics import accuracy_score   ## to calculate accuracy score


# In[46]:


Logistic_Acc =  accuracy_score(Y_test,y_pred_logistic)  #Y_test is original value for X_test
SVM_Acc = accuracy_score(Y_test, y_pred_svm)

KNeighbors_Acc = accuracy_score(Y_test,y_pred_kneighbors)


# In[47]:


# Logistic_Acc


# In[48]:


# SVM_Acc


# Accuracy of SVM model > Accuracy of Logistic Model

# ###### see this Logistic regression model is giving 83% accuracy  where as SVM mode is giving accuracy of 86%
# similarily we have to perform for other machine learning algorithms to test their accuracy...  
# now we will use support vector machine model  
#    
# Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.
# 
# The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.
# 
# SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine.

# ##### testing for kneighbors accuracy

# In[49]:


KNeighbors_Acc = accuracy_score(Y_test,y_pred_kneighbors)


# In[50]:


# KNeighbors_Acc


# ###### seee far now we are getting maximum accuracy from SVM model

# ##### tetsing accuracy for decision tree model

# In[51]:


model_tree_Acc = accuracy_score(Y_test,y_pred_tree)


# In[52]:


# model_tree_Acc


#    

# ##### testing accuracy for random forest model

# In[53]:


Random_Acc = accuracy_score(Y_test,y_pred_random)
# Random_Acc


#    
#       
#         

#    
#   
#    
#    

#  #### testing accuracy for xg model

# In[54]:


Xg_Acc = accuracy_score(Y_test,y_pred_xg)


# In[55]:


print(Xg_Acc)


#   

#   

# we can also increase the accuraccy for random forest , lets change the n_estimators whose default value is 100, changing it will take more computation and may take more time

# ##### again training model with random forst with n_estimators of 200

# In[56]:


model_random = RandomForestClassifier(n_estimators=200)


# In[57]:


model_random.fit(X_train,Y_train)


# In[58]:


y_pred_random  = model_random.predict(X_test)


# Now testing the accuracy

# In[59]:


Random_Acc = accuracy_score(Y_test,y_pred_random)
# Random_Acc


#     SEE THE ACCURACY IS INCREASED

# ##### comparing accuracy of all the models

# In[60]:


print(Logistic_Acc)
print(SVM_Acc)
print(KNeighbors_Acc)
print(model_tree_Acc)
print(Random_Acc)
print(Xg_Acc)


# ###### according to the above accuracy here we will chose SVM , bcz we find it best suitable with best accuracy

# now we will plot the bargraph

# ##### Plotting Accuracy of different ML Algorithm

# In[61]:


plt.figure(figsize=(10, 6))
plt.bar("Logistic Regression",Logistic_Acc,width=0.5)
plt.bar("KNeighbors",KNeighbors_Acc,width=0.5)
plt.bar("Support Vector Machine",SVM_Acc,width=0.5)
plt.bar("Decision Tree",model_tree_Acc,width=0.5)
plt.bar("Random Forest",Random_Acc,width=0.5)
plt.bar("XGBoost",Xg_Acc,width=0.5)
plt.xlabel("Machine Learning Algorithm")
plt.xticks(rotation=45)
plt.ylabel("Accuracy")


# #### Single Prediction

# In[62]:


dataset.head(7)


# In[63]:


# dataset


# In[86]:


print(X[916])


# In[73]:


print(X[247])


# ##### Age = 50, Sex = 0 (for female 0 and for male is 1), chestpain=0(ASY)
# ##### cholestrol = 250, FastingBloodSugar= 1,  RestingECG =2 (ST),   MaxHR = 175 ,ExerciseAngina = 1(Yes),
# ##### oldpeak = 1.9 , ST_Slope = (Down) 0

# above parameters we will use for single prediction we will enter these values


result= model_svm.predict(sc.transform([[age,sex,chestPtype,restingBp,chol,FasBs,restingecg,maxhr,exerAngina,oldpeak,st_slope]]))


if result == [0]:
    print("Person is not having any heart disease")
    st.subheader("Result: Person is not having any heart disease")
else:
    print("Person is having heart disease")
    st.subheader("Result: Person is having heart disease")