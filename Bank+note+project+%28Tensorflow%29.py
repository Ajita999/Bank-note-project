
# coding: utf-8

# # Tensorflow Project 
# 
# Dataset used is [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.
# 
# The data consists of 5 columns:
# 
# * variance of Wavelet Transformed image (continuous)
# * skewness of Wavelet Transformed image (continuous)
# * curtosis of Wavelet Transformed image (continuous)
# * entropy of image (continuous)
# * class (integer)
# 
# Where class indicates whether or not a Bank Note was authentic.
# 

# ## Importing the Data
# 
# ** Using pandas to read in the bank_note_data.csv file **

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('bank_note_data.csv')


# In[3]:


data.head()


# ## EDA
# 
# just a few quick plots of the data.
# 
# 

# In[4]:


import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[5]:


sns.countplot(x='Class',data=data)


# In[6]:


sns.pairplot(data,hue='Class')


# ## Data Preparation 
# 
# 
# ### Standard Scaling
# 
# 

# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


scaler = StandardScaler()


# In[10]:


scaler.fit(data.drop('Class',axis=1))


# In[11]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# In[12]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 
# 

# In[13]:


X = df_feat


# In[14]:


y = data['Class']


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Tensorflow

# In[17]:


import tensorflow as tf


# ** Creating a list of feature column objects using tf.feature.numeric_column()**

# In[18]:


df_feat.columns


# In[19]:


image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')


# In[20]:


feat_cols = [image_var,image_skew,image_curt,entropy]


# ** Creating an object called classifier which is a DNNClassifier from learn. Setting it to have 2 classes and a [10,20,10] hidden unit layer structure:**

# In[21]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)


# In[22]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)


# In[23]:


classifier.train(input_fn=input_func,steps=500)


# ## Model Evaluation

# In[24]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[25]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[26]:


note_predictions[0]


# In[27]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[29]:


print(confusion_matrix(y_test,final_preds))


# In[30]:


print(classification_report(y_test,final_preds))

