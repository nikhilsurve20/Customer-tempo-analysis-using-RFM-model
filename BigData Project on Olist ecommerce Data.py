#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[2]:


spark = SparkSession.builder.appName('olistRFM').getOrCreate()


# In[3]:


dfOrders = spark.read.csv('olist_orders_dataset.csv', header=True, inferSchema=True)


# In[4]:


dfOrders.printSchema()


# In[5]:


dfOrdersPayment = spark.read.csv('olist_order_payments_dataset.csv', header=True, inferSchema=True)


# In[6]:


dfOrdersPayment.printSchema()


# In[7]:


dfOrders.count()


# In[8]:


dfOrdersPayment.count()


# In[9]:


dfOrders.show(5)


# In[10]:


dfOrdersPayment.show(5)


# In[11]:


df = dfOrders.join(dfOrdersPayment,on='order_id',how='inner')


# In[12]:


df.count()


# In[13]:


df.select('order_id').distinct().count()


# In[14]:


orderFrequency = df.groupBy('order_id').count().orderBy('count', ascending = False).select('count')


# In[15]:


type(orderFrequency)


# In[16]:


data = orderFrequency.collect()


# In[17]:


df = df.drop("order_delivered_carrier_date","order_approved_at","order_delivered_customer_date","order_estimated_delivery_date","payment_installments","payment_sequential")


# In[18]:


custumerOrdersCount = df.select("customer_id").groupby("customer_id").count().orderBy("count",ascending=False)


# In[19]:


custumerOrdersPayment = df.groupby("customer_id").max("payment_value")


# In[20]:


joinResult = custumerOrdersCount.join(custumerOrdersPayment, on ="customer_id", how="inner")


# In[21]:


joinResult=joinResult.withColumnRenamed('count','frequency')


# In[22]:


import numpy as np
import datetime
from pyspark.sql.functions import year, month, dayofmonth

df = df.withColumn("order_purchase_timestamp",df["order_purchase_timestamp"].cast("double"))
OrderRecent = df.groupBy("customer_id").max("order_purchase_timestamp")
OrderRecent=OrderRecent.withColumn("max(order_purchase_timestamp)",OrderRecent["max(order_purchase_timestamp)"].cast("timestamp"))
OrderRecent = OrderRecent.withColumnRenamed("max(order_purchase_timestamp)","time_stamp")
OrderRecent.show()


# In[23]:


OrderRecent.printSchema()


# In[24]:


joinResult.printSchema()


# In[38]:


finalDf = joinResult.join(OrderRecent, on = 'customer_id')


# In[39]:


finalDf.printSchema()


# In[40]:


finalDf = finalDf.withColumnRenamed('max(payment_value)','monetary')
finalDf = finalDf.withColumnRenamed('time_stamp','recency')


# In[41]:


finalDf.printSchema()


# In[42]:


from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors


# In[43]:


finalDf.select('recency').show(1)


# In[44]:


finalDf


# In[45]:


finalDf.head()


# In[46]:


def returnEpoch(dt):
    return (dt - datetime.datetime(1970, 1, 1,0,0,0)).total_seconds()
def returnRows(x):
    return float(x[1]),float(x[2]),float(returnEpoch(x[3])),str(x[0])


# In[47]:


finalDfrdd = finalDf.rdd.map(lambda x: returnRows(x))


# In[48]:


finalDfrdd.take(3)


# In[50]:


finalDf = finalDfrdd.toDF()


# In[52]:


finalDf.show(1)


# In[55]:


finalDf = finalDf.withColumnRenamed('_1','frequency')
finalDf = finalDf.withColumnRenamed('_2','monetary')
finalDf = finalDf.withColumnRenamed('_3','recency')
finalDf = finalDf.withColumnRenamed('_4','customer_id')


# In[56]:


finalDf.head()


# In[57]:


finalDf.describe().show()


# In[58]:


clusterData = finalDf.select('frequency','monetary','recency')


# In[60]:


clusterData.head()


# In[61]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[62]:


clusterData.columns


# In[63]:


vec_assembler = VectorAssembler(inputCols = clusterData.columns, outputCol='features')


# In[65]:


final_data = vec_assembler.transform(clusterData)


# In[67]:


final_data.head()


# In[69]:


#Scaling


# In[70]:


from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)


# In[71]:


# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(final_data)


# In[72]:


# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(final_data)


# In[73]:


#Train the Model and Evaluate


# In[74]:


# Trains a k-means model.
kmeans = KMeans(featuresCol='scaledFeatures',k=3)
model = kmeans.fit(final_data)


# In[75]:


# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = " + str(wssse))


# In[76]:


# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[78]:


model.transform(final_data).select('prediction').distinct().show()


# In[ ]:




