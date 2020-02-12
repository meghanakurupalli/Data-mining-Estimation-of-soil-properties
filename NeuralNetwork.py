
# coding: utf-8

#Important note: To run the following python file,all the listed packages must be installed
#Tensorflow,Keras,Pandas,Matplotlib,Seaborn,Numpy,SciPy,scikit-learn


#Importing Python Packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


#Reading the CSV as a dataframe
df=pd.read_csv('data2.csv')


# In[18]:


df.head(2)


# In[19]:


#Slicing target feature(y),and training features(X)
y=df.iloc[:,13]


# In[20]:


X=df.iloc[:,14:]


# In[21]:


#data =pd.concat([X, y], axis=1)
count=list(range(350,2501))
count=pd.DataFrame(data=count,columns=['Wavelength'])


# In[22]:


X.head(5)


# In[23]:


X1=X.iloc[50]
X1=X1.values
X1=pd.DataFrame(data=X1,columns=['Reflection'])
X1 =pd.concat([count, X1], axis=1)
X1.head()


# In[24]:


#Regression plot
sns.set(style="darkgrid", color_codes=True)
sns.set(font_scale=1.5)
g = sns.jointplot(x='Wavelength',y='Reflection',data=X1, kind="reg", color="r", size=8)


# In[25]:


y.head(5)


# In[26]:


#Normalising data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
XS=scaler.fit_transform(X)
YS=scaler.fit_transform(y.values.reshape(-1,1))


# In[27]:


Xn = pd.DataFrame(XS, columns = X.columns)
Yn = pd.DataFrame(YS, columns = ['OC'])


# In[28]:


Xn.head()


# In[29]:


Yn.head()


# In[30]:


X=np.array(Xn)


# In[31]:


y=np.array(Yn)


# In[32]:


#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=30)
X=pca.fit_transform(X)


# In[33]:


X.shape


# In[34]:


y.shape


# In[35]:


#Train/Test split
from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[37]:


X_train.shape


# In[38]:


y_train.shape


# In[41]:


#Importing Keras,Tensorboard log,Model creation/compilation/evaluation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend
from keras import regularizers
from keras.layers import Dropout, Activation
keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

adam=keras.optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Create model
model = Sequential()
model.add(Dense(900,input_dim=30,activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu',))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu',))
model.add(Dense(1, activation='linear'))

model.summary()
# Compile mode
model.compile(loss='mse', optimizer=adam, metrics=[rmse])
# Fit the model
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, y_train, epochs=10, batch_size=1024,verbose=2,validation_split=0.1,callbacks=[tbCallBack])

scores = model.evaluate(X_test, y_test)
print ('\n Test set RMSE=%f' % scores[0])
