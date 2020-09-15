#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import pandas as pd
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


# In[3]:


iris = datasets.load_iris()


# In[4]:


x_test=pd.DataFrame(iris['data'], columns=iris['feature_names'])


# In[5]:


#x_test


# In[6]:


x_test.plot.scatter(x='petal length (cm)', y='petal width (cm)')


# In[6]:


from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()


# In[7]:


standardScaler.fit(x_test)
standardScaler.mean_
standardScaler.scale_ 


# In[8]:


# 1.min_max歸一化 
#min_max_scaler = preprocessing.MinMaxScaler()
#minmax_x = min_max_scaler.fit_transform(x_test)
#print (minmax_x)
#minmax_x.plot.scatter(x='sepal length (cm)', y ='sepal width (cm)')
#x_quantum1=(minmax_x)*(math.pi)
#print(x_quantum1)
#plt.scatter(x_quantum1[:,2],x_quantum1[:,3])
#x_quantum1.plot.scatter(x='sepal length (cm)',y='sepal width (cm)')


# In[9]:


# 2. Z-score 歸一化


# In[10]:


#所有x_test的算術平均數
mu = x_test.mean()
#標準差
std = x_test.std()
#標準化後之結果
z_score_normalized = (x_test - mu) / std
#最大-最小
print(z_score_normalized)


# In[11]:


#plt.scatter(z_score_normalized[:,0],z_score_normalized[:,1])


# In[12]:


#print(z_score_normalized["sepal length (cm)"].min)


# In[13]:


#轉pi
x_quantum2=z_score_normalized*(math.pi)
print(x_quantum2)


# In[14]:


x_quantum2.plot.scatter(x='sepal length (cm)',y='sepal width (cm)')


# In[15]:


x_quantum2.plot.scatter(x='petal length (cm)', y='petal width (cm)')


# In[16]:


#轉回角度 
list2_ft1=x_quantum2["sepal length (cm)"]
print (list2_ft1)


# In[23]:


#len(list2_ft1)


# In[17]:


x_angle2_ft1=[]
for i in range (0,len(list2_ft1)):
    x_angle2_ft1.append(math.degrees((list2_ft1[i])))


# In[18]:


#x_angle2_ft1


# In[19]:


#看feature1的分布
X1=range(len(x_angle2_ft1))
Y1=x_angle2_ft1
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[20]:


X2=range(len(list2_ft1))
Y2=list2_ft1
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[21]:


X3=range(len(x_test["sepal length (cm)"]))
Y3=x_test["sepal length (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal length')


# In[22]:


list2_ft2=x_quantum2["sepal width (cm)"]
print (list2_ft2)


# In[23]:


x_angle2_ft2=[]
for i in range (0,len(list2_ft2)):
    x_angle2_ft2.append(math.degrees((list2_ft2[i])))


# In[24]:


#x_angle2_ft2


# In[25]:


X1=range(len(x_angle2_ft2))
Y1=x_angle2_ft2
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[26]:


X2=range(len(list2_ft2))
Y2=list2_ft2
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[27]:


X3=range(len(x_test["sepal width (cm)"]))
Y3=x_test["sepal width (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='sepal width')


# In[28]:


list2_ft3=x_quantum2["petal length (cm)"]
print (list2_ft3)


# In[29]:


x_angle2_ft3=[]
for i in range (0,len(list2_ft3)):
    x_angle2_ft3.append(math.degrees((list2_ft3[i])))


# In[30]:


#x_angle2_ft3


# In[31]:


X1=range(len(x_angle2_ft3))
Y1=x_angle2_ft3
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[32]:


X2=range(len(list2_ft3))
Y2=list2_ft3
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[33]:


X3=range(len(x_test["petal length (cm)"]))
Y3=x_test["petal length (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal length')


# In[34]:


list2_ft4=x_quantum2["petal width (cm)"]
print (list2_ft4)


# In[35]:


x_angle2_ft4=[]
for i in range (0,len(list2_ft4)):
    x_angle2_ft4.append(math.degrees((list2_ft4[i])))


# In[36]:


#x_angle2_ft4


# In[37]:


X1=range(len(x_angle2_ft4))
Y1=x_angle2_ft4
plt.bar(X1, Y1, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal width')


# In[38]:


X2=range(len(list2_ft4))
Y2=list2_ft4
plt.bar(X2, Y2, width = 3, facecolor = 'blue', edgecolor = 'white', label='petal width')


# In[39]:


X3=range(len(x_test["petal width (cm)"]))
Y3=x_test["petal width (cm)"]
plt.bar(X3, Y3, width = 3, facecolor = 'blue', edgecolor = 'white', label='width length')


# In[40]:


# U3 gate 電路 完成data loader


# In[41]:


from sklearn.utils import shuffle


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


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


# In[44]:


#x_test_tra


# In[45]:


iris_test_data=iris.data
iris_test_label=iris.target
#print(iris_test_label)


# In[47]:


train_data , test_data , train_label , test_label = train_test_split(iris_test_data,iris_test_label,test_size=0.2,random_state=6666)


# In[48]:


#print(train_data.shape) 
#print(test_data.shape) 
#print(train_label.shape) 
#print(test_label.shape)


# In[49]:


knn1 = KNeighborsClassifier()


# In[50]:


knn1.fit(train_data,train_label)


# In[51]:


print(knn1.predict(test_data))
print(test_label)


# In[52]:


#測試KNN演算法的好壞
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_label,knn1.predict(test_data)))


# In[54]:


#x_quantum2


# In[55]:


qiris_test_data=x_quantum2
qiris_test_label=iris.target


# In[56]:


qtrain_data , qtest_data , qtrain_label , qtest_label = train_test_split(qiris_test_data,qiris_test_label,test_size=0.2,random_state=1666)


# In[57]:


knn2 = KNeighborsClassifier()


# In[58]:


knn2.fit(qtrain_data,qtrain_label)


# In[59]:


print(knn2.predict(qtest_data))
print(qtest_label)


# In[60]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(qtest_label,knn2.predict(qtest_data)))


# In[62]:


#歸一化後角度與原始DATA比較
print(knn1.predict(test_data))
print(test_label)
print(knn2.predict(qtest_data))
print(qtest_label)


# In[63]:


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit,Aer, execute
from math import sqrt,cos,sin,pi,acos


# In[64]:


#shuffle&切割資料train&test
qiris_test_data=x_quantum2
qiris_test_label=iris.target
qtrain_data , qtest_data , qtrain_label , qtest_label = train_test_split(qiris_test_data,qiris_test_label,test_size=0.2,random_state=1666)
#prparing training data in quantum state
qtrain_data_ft1=[]
qtrain_data_ft1=qtrain_data['sepal length (cm)']
qtrain_data_ft2=[]
qtrain_data_ft2=qtrain_data['sepal width (cm)']
qtrain_data_ft3=[]
qtrain_data_ft3=qtrain_data['petal width (cm)']
qtrain_data_ft4=[]
qtrain_data_ft4=qtrain_data['petal width (cm)']


# In[65]:


#preparing testing data in quantum state
qtest_data_ft1=[]
qtest_data_ft1=qtest_data['sepal length (cm)']
qtest_data_ft2=[]
qtest_data_ft2=qtest_data['sepal width (cm)']
qtest_data_ft3=[]
qtest_data_ft3=qtest_data['petal width (cm)']
qtest_data_ft4=[]
qtest_data_ft4=qtest_data['petal width (cm)']


# In[66]:


from qiskit.quantum_info import state_fidelity


# In[67]:


device=Aer.get_backend('qasm_simulator')
q=QuantumRegister(5)
c=ClassicalRegister(5)
cir1=QuantumCircuit(q,c)
cir1.u3(qtrain_data_ft1[2],qtrain_data_ft2[2],0,q[1]) #quantum state training data ft1&ft2
cir1.u3(qtrain_data_ft3[2],qtrain_data_ft4[2],0,q[2]) #quantum state training data ft3&ft4
cir1.u3(qtest_data_ft1[3],qtest_data_ft2[3],0,q[3]) #quantum state testing data ft1&ft2
cir1.u3(qtest_data_ft3[3],qtest_data_ft4[3],0,q[4]) #quantum state testing data ft3&ft4
cir1.barrier()
cir1.h(q[0])
cir1.cswap(q[0],q[1],q[3])
cir1.cswap(q[0],q[2],q[4])
cir1.h(q[0])
cir1.measure(q[0],c[0])
cir1.draw(output='mpl')


# In[68]:


backend = Aer.get_backend('qasm_simulator')
job = execute(cir1, backend)
result = job.result()


# In[69]:


#丟全部data 找K minimum distance


# In[70]:


from random import choice


# In[71]:


#index reset
a=qtrain_data_ft1.reset_index(drop=True)
b=qtrain_data_ft2.reset_index(drop=True)
c=qtrain_data_ft3.reset_index(drop=True)
d=qtrain_data_ft4.reset_index(drop=True)


# In[72]:


#index reset
w=qtest_data_ft1.reset_index(drop=True)
x=qtest_data_ft2.reset_index(drop=True)
y=qtest_data_ft3.reset_index(drop=True)
z=qtest_data_ft3.reset_index(drop=True)


# In[73]:


#data_loader
Fidelity=[]
device=Aer.get_backend('qasm_simulator')
for i in range(len(a)):
    for j in range(len(w)):
        q_t=QuantumRegister(5)
        c_t=ClassicalRegister(5)
        cir2=QuantumCircuit(q_t,c_t)
        cir2.u3(a[i],b[i],0,q_t[1])
        cir2.u3(c[i],d[i],0,q_t[2])
        cir2.u3(w[j],x[j],0,q_t[3])
        cir2.u3(y[j],z[j],0,q_t[4])
        cir2.barrier()
        cir2.h(q_t[0])
        cir2.cswap(q_t[0],q_t[1],q_t[3])
        cir2.cswap(q_t[0],q_t[2],q_t[4])
        cir2.h(q_t[0])
        cir2.barrier()
        cir2.measure(q_t[0],c_t[0])
        backend = Aer.get_backend('qasm_simulator')
        job = execute(cir2, backend, shots=1024)
        result = job.result()
        temp=result.get_counts()
        Fidelity.append(temp)
#cir2.draw(output='mpl')


# In[74]:


Fidelity


# In[95]:


#cir2.draw(output='mpl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




