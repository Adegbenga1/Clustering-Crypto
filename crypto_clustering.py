#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[149]:


conda install -c pyviz hvplot


# In[150]:


# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ### Deliverable 1: Preprocessing the Data for PCA

# In[151]:


# Load the crypto_data.csv dataset.
file_path = r"C:\Users\AYOOLA5\Desktop\University of Toronto\MODULE 18\Module 18 Challenge/crypto_data.csv"
crypto_df = pd.read_csv(file_path, index_col=0)
crypto_df.head(10)


# In[152]:


for column in crypto_df.columns:
    print(f"Column {column} has {crypto_df[column].isnull().sum()} null values")


# In[153]:


# Keep all the cryptocurrencies that are being traded.
crypto_df= crypto_df[crypto_df['IsTrading'] ==True]


# In[154]:


print(crypto_df.shape)
crypto_df.head(10)


# In[155]:


# Keep all the cryptocurrencies that have a working algorithm.
crypto_df.Algorithm.value_counts().head()


# In[185]:


# Remove the "IsTrading" column. 
cryptorevised_df = crypto_df.drop(['IsTrading'], axis=1)
cryptorevised_df.head()


# In[189]:


# Remove rows that have at least 1 null value.
cryptorevised_df.dropna(axis=0,how="any")


# In[190]:


# Keep the rows where coins are mined.
cryptorevised_df = cryptorevised_df[cryptorevised_df["TotalCoinsMined"]>0]
cryptorevised_df


# In[191]:


# Create a new DataFrame that holds only the cryptocurrencies names.
cryptoname_df = pd.DataFrame(
    data=cryptorevised_df, columns=["CoinName"])
cryptoname_df.head()


# In[192]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
cryptorevised_df = cryptorevised_df.drop("CoinName", axis=1)
cryptorevised_df.head(10)


# In[193]:


# Use get_dummies() to create variables for text features.                              
df = pd.get_dummies(cryptorevised_df,columns = ['Algorithm', 'ProofType'])
df.dropna()


# In[199]:


# Standardize the data with StandardScaler()
X_scaled = StandardScaler().fit_transform(df)
print(X_scaled [0:5])


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[231]:


# Using PCA to reduce dimension to three principal components.
pca = PCA(n_components=3)
X_scaled_transformed = pca.fit_transform(X_scaled)
print(X_scaled_transformed)


# In[232]:


# Create a DataFrame with the three principal components.
df_pca = pd.DataFrame(
    data = X_scaled_transformed, columns = ["PC 1", "PC 2", "PC 3"],index=cryptorevised_df.index
)

df_pca


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[233]:


# Create an elbow curve to find the best value for K.
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i, random_state=1)
    km.fit(df_pca)
    inertia.append(km.inertia_)

# Create the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")


# Running K-Means with `k=4`

# In[234]:


# Initialize the K-Means model.
model = KMeans(n_clusters=4, random_state=1)
# Fit the model
model.fit(df_pca)
# Predict clusters
predictions = model.predict(df_pca)
print(predictions)
df_pca["Class"] = model.labels_


# In[235]:


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
df_pca["Class"] = model.labels_
df_pca.head()


# In[236]:


df_pca["Class"].value_counts()


# In[237]:


clustered_df = cryptorevised_df.join(df_pca, how='inner')
clustered_df.head(10)


# In[238]:


# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = pd.concat([cryptorevised_df,df_pca],axis=1,sort = False)
clustered_df


# In[241]:


#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df = clustered_df.join(crypto_df["CoinName"], how='inner')
clustered_df.head()


# In[244]:


# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[245]:


# Creating a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_df, 
    x="PC 1", 
    y="PC 2", 
    z="PC 3", 
    color="Class", 
    symbol="Class", 
    hover_name="CoinName", 
    hover_data=["Algorithm", "TotalCoinsMined", "TotalCoinSupply"])
fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# In[246]:


# Create a table with tradable cryptocurrencies.
clustered_df.hvplot.table(columns=['CoinName', 'Algorithm', 'ProofType', 'TotalCoinsMined', 'TotalCoinSupply', 'Class'], sortable=True, selectable=True)


# In[260]:


# Print the total number of tradable cryptocurrencies.
clustered_df['CoinName'].count()


# In[261]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
cluster_df = clustered_df[['TotalCoinSupply', 'TotalCoinsMined']]
X_minmax = MinMaxScaler().fit_transform(cluster_df)
X_minmax


# In[262]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
New_df = pd.DataFrame(
    data = X_minmax, columns=["TotalCoinSupply_scaled", "TotalCoinsMined_scaled"], index=cryptorevised_df.index)
New_df


# In[268]:


# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
New_df["CoinName"] = clustered_df["CoinName"]
New_df


# In[271]:


# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
New_df["Class"] = clustered_df["Class"]
New_df.head(10)


# In[274]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
New_df.hvplot.scatter(x="TotalCoinsMined_scaled", y="TotalCoinSupply_scaled", by="Class",
                          xlabel="Total Cryptocurrency Coins Mined",
                          ylabel="Total Cryptocurrency Coin Supply",
                          hover_cols = ["CoinName"]
                          )

