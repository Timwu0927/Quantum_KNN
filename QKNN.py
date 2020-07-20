#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
# Loading your IBM Q account(s)
provider = IBMQ.load_account()
from sklearn import preprocessing
import numpy as np
import math


# In[8]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import pandas as pd
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


# In[9]:


iris = datasets.load_iris()


# In[10]:


x_test=pd.DataFrame(iris['data'], columns=iris['feature_names'])


# In[11]:


x_test


# In[12]:


x_test.plot.scatter(x='petal length (cm)', y='petal width (cm)')


# In[13]:


#color={
    #0:'r',
    #1:'g',
    #2:'b'
#}


# In[14]:


#x_test['color']=x_test['target'].map(color)


# In[15]:


from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()


# In[16]:


standardScaler.fit(x_test)
standardScaler.mean_
standardScaler.scale_ 


# In[17]:


# 1.min_max歸一化 
#min_max_scaler = preprocessing.MinMaxScaler()
#minmax_x = min_max_scaler.fit_transform(x_test)
#print (minmax_x)


# In[18]:


#minmax_x.plot.scatter(x='sepal length (cm)', y ='sepal width (cm)')


# In[19]:


#x_quantum1=(minmax_x)*(math.pi)
#print(x_quantum1)


# In[20]:


#plt.scatter(x_quantum1[:,2],x_quantum1[:,3])
#x_quantum1.plot.scatter(x='sepal length (cm)',y='sepal width (cm)')


# In[21]:


# 2. Z-score 歸一化


# In[112]:


#所有x_test的算術平均數
mu = x_test.mean()
#標準差
std = x_test.std()
#標準化後之結果
z_score_normalized = (x_test - mu) / std
#最大-最小
print(z_score_normalized)


# In[23]:


#plt.scatter(z_score_normalized[:,0],z_score_normalized[:,1])


# In[24]:


#print(z_score_normalized["sepal length (cm)"].min)


# In[113]:


x_quantum2=z_score_normalized*(math.pi)
print(x_quantum2)


# In[26]:


x_quantum2.plot.scatter(x='sepal length (cm)',y='sepal width (cm)')


# In[27]:


x_quantum2.plot.scatter(x='petal length (cm)', y='petal width (cm)')


# In[28]:


#轉回角度 
list2_ft1=x_quantum2["sepal length (cm)"]
print (list2_ft1)


# In[29]:


#len(list2_ft1)


# In[30]:


x_angle2_ft1=[]
for i in range (0,len(list2_ft1)):
    x_angle2_ft1.append(math.degrees((list2_ft1[i])))


# In[31]:


#x_angle2_ft1


# In[32]:


#看feature1的分布
X1=range(len(x_angle2_ft1))
Y1=x_angle2_ft1
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[33]:


X2=range(len(list2_ft1))
Y2=list2_ft1
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[34]:


X3=range(len(x_test["sepal length (cm)"]))
Y3=x_test["sepal length (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[35]:


list2_ft2=x_quantum2["sepal width (cm)"]
print (list2_ft2)


# In[36]:


x_angle2_ft2=[]
for i in range (0,len(list2_ft2)):
    x_angle2_ft2.append(math.degrees((list2_ft2[i])))


# In[37]:


#x_angle2_ft2


# In[38]:


X1=range(len(x_angle2_ft2))
Y1=x_angle2_ft2
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[39]:


X2=range(len(list2_ft2))
Y2=list2_ft2
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[40]:


X3=range(len(x_test["sepal width (cm)"]))
Y3=x_test["sepal width (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[41]:


list2_ft3=x_quantum2["petal length (cm)"]
print (list2_ft3)


# In[42]:


x_angle2_ft3=[]
for i in range (0,len(list2_ft3)):
    x_angle2_ft3.append(math.degrees((list2_ft3[i])))


# In[117]:


#x_angle2_ft3


# In[44]:


X1=range(len(x_angle2_ft3))
Y1=x_angle2_ft3
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[45]:


X2=range(len(list2_ft3))
Y2=list2_ft3
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[46]:


X3=range(len(x_test["petal length (cm)"]))
Y3=x_test["petal length (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[47]:


list2_ft4=x_quantum2["petal width (cm)"]
print (list2_ft4)


# In[48]:


x_angle2_ft4=[]
for i in range (0,len(list2_ft4)):
    x_angle2_ft4.append(math.degrees((list2_ft4[i])))


# In[49]:


#x_angle2_ft4


# In[50]:


X1=range(len(x_angle2_ft4))
Y1=x_angle2_ft4
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal width')


# In[51]:


X2=range(len(list2_ft4))
Y2=list2_ft4
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal width')


# In[52]:


X3=range(len(x_test["petal width (cm)"]))
Y3=x_test["petal width (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='width length')


# In[53]:


# U3 gate 電路 完成data loader


# In[54]:


from sklearn.utils import shuffle


# In[55]:


def dataloader():
    #shuffle&切割資料train&test
    qiris_test_data=x_quantum2
    qiris_test_label=iris.target
    qtrain_data , qtest_data , qtrain_label , qtest_label = train_test_split(qiris_test_data,qiris_test_label,test_size=0.2,random_state=1666)
    #test&train找similarity
    qtrain_data


# In[187]:


def circ(N):
    q=QuantumRegister(N)
    c=ClassicalRegister(N)
    cir=QuantumCircuit(q,c)
    


# In[189]:


#cir=circ(2)


# In[190]:


#cir.draw(output='mpl')


# In[ ]:


device=Aer.get_backend('qasm_simulator')
prob=execute(cir,device,shots=1024)
print(prob.result().get_counts(cir))


# In[56]:


from sklearn.model_selection import train_test_split


# In[166]:


#****用角度做古典KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#吃資料
iris_test = datasets.load_iris()
x_test_tra=pd.DataFrame(iris_test['data'],iris_test['target'])
#從k=1開始測試


# In[191]:


#x_test_tra


# In[168]:


iris_test_data=iris.data
iris_test_label=iris.target
#print(iris_test_label)


# In[169]:


train_data , test_data , train_label , test_label = train_test_split(iris_test_data,iris_test_label,test_size=0.2,random_state=6666)


# In[170]:


#print(train_data.shape) 
#print(test_data.shape) 
#print(train_label.shape) 
#print(test_label.shape)


# In[171]:


knn1 = KNeighborsClassifier()


# In[172]:


knn.fit(train_data,train_label)


# In[173]:


print(knn.predict(test_data))
print(test_label)


# In[174]:


#測試KNN演算法的好壞
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_label,knn.predict(test_data)))


# In[175]:


x_quantum2


# In[176]:


qiris_test_data=x_quantum2
qiris_test_label=iris.target


# In[177]:


qtrain_data , qtest_data , qtrain_label , qtest_label = train_test_split(qiris_test_data,qiris_test_label,test_size=0.2,random_state=1666)


# In[178]:


knn2 = KNeighborsClassifier()


# In[179]:


knn.fit(qtrain_data,qtrain_label)


# In[180]:


print(knn.predict(qtest_data))
print(qtest_label)


# In[181]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(qtest_label,knn.predict(qtest_data)))


# In[182]:


#歸一化後角度與原始DATA比較
print(knn.predict(test_data))
print(test_label)
print(knn.predict(qtest_data))
print(qtest_label)


# In[ ]:




