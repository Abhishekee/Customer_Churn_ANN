#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


# In[4]:


df = pd.read_csv("churn_rate.csv")
df.head()


# In[16]:


new_df_x = df.iloc[:,3:-1]
new_df_x


# In[21]:


X = new_df_x.values
X


# In[19]:


new_df_y = df.iloc[:,-1]
new_df_y


# In[23]:


Y = new_df_y.values
Y


# In[24]:


L = LabelEncoder()
X[:,2] = L.fit_transform(X[:,2])
print(X)


# In[25]:


ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)


# In[72]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
print(X_train)


# In[73]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[77]:


Y_train = Y_train.reshape(7000,1)
Y_train.shape

Y_test = Y_test.reshape(3000,1)
Y_test.shape


# In[78]:


X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)


# In[79]:


ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

print(X_train)


# In[82]:


ANN = tf.keras.models.Sequential()
ANN.add(tf.keras.Input(shape = (12,)))
ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
ANN.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[86]:


ANN.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = ANN.fit(X_train, Y_train, batch_size = 25, epochs = 200)


# In[90]:


import matplotlib.pyplot as plt
plt.title('model accuracy')
plt.plot(history.history['accuracy'], color = "green")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[91]:


plt.title('Model Loss')
plt.plot(history.history['loss'], color = 'red')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[93]:


print(ANN.predict(ss.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(ANN.predict(ss.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# In[96]:


y_pred = ANN.predict(X_test)
print(y_pred)


# In[98]:


classifier = (y_pred > 0.5)
classifier


# In[100]:


print(np.concatenate((classifier.reshape(len(classifier),1), Y_test.reshape(len(Y_test),1)), 1))


# In[101]:


cf = confusion_matrix(Y_test, classifier)
print("Confusion Matrix:", cf)
print("Accuracy score is:", accuracy_score(Y_test, classifier))


# In[103]:


pred = np.concatenate((classifier.reshape(len(classifier),1), Y_test.reshape(len(Y_test),1)), 1)
pd.DataFrame(pred).to_csv("Predictions")


# In[ ]:




