#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.graph_objects as go


# In[2]:


df = pd.read_csv('dataset_2019_2022.csv')


# In[3]:


df.head(5)


# In[4]:


#Question 1) What are the top 10 selling commodities by loyalty type? 
df.groupby(['commodity']).agg(total_revenue=('price',sum)) \
    .sort_values('total_revenue', ascending = False).head(10)


# In[5]:


# The top 10 commodities among loyalty types
tmp_1 = df.groupby(['loyalty', 'commodity']).agg(total_revenue=('price', sum)).reset_index()
top_10_1 = pd.concat(
    [tmp_1[tmp_1.loyalty==l] \
         .sort_values('total_revenue', ascending=False) \
     .head(10) for l in tmp_1.loyalty.unique()]).reset_index(drop=True)

tmp_1.head(10)
top_10_1.head(100)


# In[6]:


#Chart the top 10 commodities for each loyalty type
data = []
for d in top_10_1.commodity.unique():
    tmp1 = top_10_1[top_10_1.commodity==d].groupby(['loyalty']).agg(revenue=('total_revenue', sum)).reset_index()
    data.append(go.Bar(x=tmp1.loyalty, y=tmp1.revenue, name = d))
        
go.Figure(
    data = data,
    layout = go.Layout(
        title ='Top commodities per Loyalty Group',
        yaxis=dict(
            title='Revenue'
        ),
        xaxis=dict(
            title='Loyalty Type'
        )
    )
).show(renderer = 'iframe')


# In[7]:


#Question 2) What is the revenue trend by each age band?

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format="%d/%m/%Y")

# Extract year from transaction_date
df['year'] = df['transaction_date'].dt.year

# Group by age band and year, and aggregate total_revenue
tmp_stats = df.groupby(['age_band', 'year']).agg(total_revenue=('price', 'sum')).reset_index()

# Filter out data from 2022
tmp_stats_filtered = tmp_stats[tmp_stats['year'] != 2022]
tmp_stats_filtered.head(50)


# In[8]:


#plot revenue by age band
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# Loop through unique age_band
for age_band in tmp_stats_filtered['age_band'].unique():
    # Filter data for the current age_band
    data_subset = tmp_stats_filtered[tmp_stats_filtered['age_band'] == age_band]
    # Plot total revenue over the years for the current age_band
    plt.plot(data_subset['year'], data_subset['total_revenue'], label=age_band)

plt.title('Sales revenue by age band (excluding 2022)')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.legend(title='age_band', loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()


# In[9]:


#Question 3) What is the revenue trend by each household type?

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'], format="%d/%m/%Y")

# Extract year from transaction_date
df['year'] = df['transaction_date'].dt.year

# Group by household_type and year, and aggregate total_revenue
tmp_stats = df.groupby(['household_type', 'year']).agg(total_revenue=('price', 'sum')).reset_index()

# Filter out data from 2022
tmp_stats_filtered = tmp_stats[tmp_stats['year'] != 2022]
tmp_stats_filtered.head(20)


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

# Loop through unique household types
for household_type in tmp_stats_filtered['household_type'].unique():
    # Filter data for the current household type
    data_subset = tmp_stats_filtered[tmp_stats_filtered['household_type'] == household_type]
    # Plot total revenue over the years for the current household type
    plt.plot(data_subset['year'], data_subset['total_revenue'], label=household_type)

plt.title('Sales revenue by household type (excluding 2022)')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.legend(title='Household Type', loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()




# In[11]:


#Question 4) What is the revenue prediction for the next 2 years should current trends continue?

from pandas.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose #library for time series analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
statsmodels.__version__


# In[12]:


df['t_date'] = pd.to_datetime(df.transaction_date) #convert to date format
df['t_date'] = df.t_date + pd.offsets.MonthBegin(-1) #send dates to first day of the month


# In[13]:


df.head()


# In[14]:


ts = df.groupby(['t_date']).agg(total_revenue=('price', sum)).reset_index()


# In[15]:


ts.head()


# In[16]:


#Grab 3 years of data
training = ts.loc[ts.t_date < '2022-01-01'].set_index('t_date')
training.shape


# In[17]:


training.plot();


# In[18]:


# See the components of a time series
# - Observed
# - Trend (direction)
# - Seasonal (repeated pattern)
# - Residual (noise)
ts_components = seasonal_decompose(training)
ts_components.plot();


# In[19]:


# Using mean and variance to check if time series is stationary
split = round(len(training) / 2)
x1 = training[0:split]
x2 = training[split:]
mean1= x1.mean()
mean2= x2.mean()
print("Mean 1 & 2= ", mean1[0], mean2[0])
var1=x1.var()
var2=x2.var()
print("Variance 1 & 2= ",var1[0], var2[0])


# In[20]:


#Means are around the same value, but variances seem to be in different ranges.We can use a statistical test to test whether the time series is stationary. 
# We use the Augmented Dickey-Fuller test
test_adf = adfuller(training)
print('ADF test = ', test_adf[0])
print('p-value = ', test_adf[1])


# In[21]:


#Given that the ADF value is negative and p-value < 0.05, we can reject the null hyphotesis and tell that our time series is stationary. Now we can apply a forecasting method. 
autocorrelation_plot(training);


# In[22]:


# fit the model
model = ARIMA(training, order=(3,0,0), freq='MS')
model_fit = model.fit()


# In[23]:


# Test dataset
test = ts.loc[ts.t_date >= '2022-01-01'].set_index('t_date').reset_index()
test.head(10)


# In[24]:


whole = ts.set_index('t_date').squeeze().copy()
history = whole.take(range(36))
future = test.set_index('t_date').squeeze().copy()
for t in range(len(future)):
    model = ARIMA(history, order=(3,0,0), freq='MS')
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    obs = future[t]
    history = whole.take(range(36 + t+1))
    print('prediction', yhat, ', expected', obs)


# In[25]:


# Forecasted revenue for the next 24 months

model = ARIMA(history, order=(3, 0, 0), freq='MS')
model_fit = model.fit()
output = model_fit.forecast(steps=24)

print("Forecasted revenue for the next 24 months:")
for i, yhat in enumerate(output):
    print(f"Month {i+1}: {yhat}")


# In[26]:


# Slice the output array for months 1 to 12 and calculate the sum
revenue_months_1_to_12 = sum(output[:12])

# Slice the output array for months 13 to 24 and calculate the sum
revenue_months_13_to_24 = sum(output[12:])

print("Forecasted revenue for 2022:", revenue_months_1_to_12)
print("Forecasted revenue for 2023:", revenue_months_13_to_24)

