#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd

melbourne_file_path = '/Users/devanhall/Downloads/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.describe()

melbourne_data.columns

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

X.describe()

X.head()


from sklearn.tree import DecisionTreeRegressor

# now we define a model. Ensure to specifiy a number for the random_state for same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# now fit the model
melbourne_model.fit(X, y)

print('Making prediction for the following 5 rows...')
print(X.head())
print('The predictions are: ')
print(melbourne_model.predict(X.head()))





# In[23]:


melbourne_data.head()


# In[26]:


from sklearn.metrics import mean_absolute_error


# In[31]:


predicted_melbourne_prices = melbourne_model.predict(X)


# In[32]:


mean_absolute_error(y, predicted_melbourne_prices)


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# def model

melbourne_model = DecisionTreeRegressor()

# fit model

melbourne_model.fit(train_X, train_y)


# In[36]:


# now get predicted prices on dataset
val_predictions = melbourne_model.predict(val_X)


# In[37]:


print(mean_absolute_error(val_y, val_predictions))


# In[40]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[46]:


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print('Max leaf node: %d \t\t Mean Absolute Error: %d' %(max_leaf_nodes, my_mae))


# In[47]:


from sklearn.ensemble import RandomForestRegressor


# In[48]:


from sklearn.metrics import mean_absolute_error


# In[50]:


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)


# In[51]:


print(mean_absolute_error(val_y, melb_preds))


# In[ ]:




