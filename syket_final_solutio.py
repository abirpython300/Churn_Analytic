#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle


# In[3]:


df = pd.read_csv("D:\IVY PRO SCHOOL\python\syket intership\Telco.csv")


# In[4]:


df.shape


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.shape


# In[11]:


df.columns.str.lower().inplace=True


# In[12]:


df.columns.str.lower()


# In[13]:


df.info()


# In[14]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# In[15]:


df.columns


# In[17]:


df.rename(columns={
    'seniorcitizen': 'senior_citizen',
    'phoneservice': 'phone_service',
    'onlinesecurity': 'online_security',
    'onlinebackup': 'online_backup',
    'multiplelines': 'multiple_lines',
    'internetservice': 'internet_service',
    'deviceprotection': 'device_protection',
    'techsupport': 'tech_support',
    'streamingtv': 'streaming_tv',
    'streamingmovies': 'streaming_movies',
    'paymentmethod': 'payment_method',
    'paperlessbilling': 'paperless_billing',
    'monthlycharges': 'monthly_charges',
    'totalcharges': 'total_charges'
}, inplace=True)


# In[18]:


df.info()


# In[19]:


df.head(10)


# In[20]:


df.describe()


# In[31]:


print(df["churn"].value_counts())


# In[ ]:





# In[25]:


len(df[df["total_charges"]== " "])


# In[26]:


df["total_charges"] = df["total_charges"].replace(" ", "0.0")


# In[27]:


len(df[df["total_charges"]== " "])


# In[28]:


df["total_charges"] = df["total_charges"].astype(float)


# In[33]:


numeric_summary = df[["tenure", "monthly_charges", "total_charges"]].describe()


# In[34]:


plt.figure(figsize=(16, 5))


# In[35]:


plt.subplot(1, 3, 1)
sns.countplot(data=df, x="churn", palette="pastel")
plt.title("churn_distribution")


# In[36]:


#tenure_distribution
plt.subplot(1, 3, 2)
sns.histplot(df["tenure"], bins=30, kde=True, color="skyblue")
plt.title("Tenure Distribution")


# In[37]:


#monthly charge distributio
plt.subplot(1, 3, 3)
sns.histplot(df["monthly_charges"], bins=30, kde=True, color="orange")
plt.title("Monthly Charges Distribution")


# In[40]:


# Calculate churn rate by categorical features
categorical_cols = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "internet_service", "contract", "payment_method",
    "paperless_billing"
]
churn_rate_summary = {}

for col in categorical_cols:
    churn_rate_summary[col] = (
        df.groupby(col)["churn"]
        .value_counts(normalize=True)
        .rename("Proportion")
        .mul(100)
        .reset_index()
    )


# In[41]:


# Plot churn rates for key features
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.barplot(
        data=churn_rate_summary[col],
        x=col, y="Proportion", hue="churn", palette="pastel", ax=axes[i]
    )
    axes[i].set_title(f"Churn by {col}")
    axes[i].set_ylabel("% Customers")
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

churn_rate_summary.keys()


# In[42]:


# Clean TotalCharges (convert to numeric, coerce errors)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df = df.dropna(subset=["total_charges"])


# In[43]:


# Define categorical columns for churn analysis
categorical_cols = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "internet_service", "contract", "payment_method",
    "paperless_billing"
]

# Calculate churn proportions
churn_rate_summary = {}
for col in categorical_cols:
    churn_rate_summary[col] = (
        df.groupby(col)["churn"]
        .value_counts(normalize=True)
        .rename("Proportion")
        .mul(100)
        .reset_index()
    )


# In[44]:


# Plot churn rates for key features
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.barplot(
        data=churn_rate_summary[col],
        x=col, y="Proportion", hue="churn", palette="pastel", ax=axes[i]
    )
    axes[i].set_title(f"Churn by {col}")
    axes[i].set_ylabel("% Customers")
    axes[i].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()


# #Task 4
# Visualize customer distribution by tenure
#  using pie or donut charts. (e.g., 0-12
#  months, 13-36 months, 37+ months).
#  Use a clustered bar chart to compare
#  average monthly charges across tenure
#  categories, 
# adding annotations to highlight significant
#  trends.

# In[46]:


# Create tenure groups
bins = [0, 12, 36, df["tenure"].max()]
labels = ["0-12 months", "13-36 months", "37+ months"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)


# In[47]:


tenure_distribution = df["tenure_group"].value_counts().sort_index()


# In[48]:


# Plot donut chart
plt.figure(figsize=(7,7))
colors = sns.color_palette("pastel")[0:3]
plt.pie(
    tenure_distribution, 
    labels=tenure_distribution.index, 
    autopct="%1.1f%%", 
    startangle=90, 
    colors=colors, 
    wedgeprops={'width':0.4}  # donut effect
)


# In[51]:


# Calculate average monthly charges by tenure group and churn status
avg_charges = (
    df.groupby(["tenure_group", "churn"])["monthly_charges"]
    .mean()
    .reset_index()
)

# Plot clustered bar chart
plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=avg_charges, 
    x="tenure_group", 
    y="monthly_charges", 
    hue="churn", 
    palette="pastel"
)
# Add annotations on bars
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f}', 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom', fontsize=9, color="black", xytext=(0,3), textcoords="offset points"
    )

plt.title("Average Monthly Charges by Tenure Group and Churn Status", fontsize=14)
plt.ylabel("Average Monthly Charges ($)")
plt.xlabel("Tenure Group")
plt.legend(title="Churn")
plt.tight_layout()
plt.show()

avg_charges


# Tasks 5:  Advanced Analysis
#  Description: 
# Perform deeper analysis by grouping customers
#  by tenure to compute statistics for charges and
#  churn.
#  Analyze churn rates by demographics (e.g.,
#  gender, senior citizen status).
#  payment methods, and contract types.
#  Visualize trends over time (if applicable) or
#  lifecycle stages to identify patterns.

# In[53]:


# Recreate tenure groups
bins = [0, 12, 36, df["tenure"].max()]
labels = ["0-12 months", "13-36 months", "37+ months"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

# Group by tenure group
tenure_stats = df.groupby("tenure_group").agg(
    Customers=("customerid", "count"),
    Avg_Monthly_Charges=("monthly_charges", "mean"),
    Avg_Total_Charges=("total_charges", "mean"),
    Churn_Rate=("churn", lambda x: (x.eq("Yes").mean() * 100))
).reset_index()

# Display results
print(tenure_stats)


# In[63]:


# Churn rate by Gender
churn_by_gender = df.groupby("gender")["churn"].value_counts(normalize=True).rename("Proportion").mul(100).reset_index()
import seaborn as sns
import matplotlib.pyplot as plt

# List of categorical variables to analyze
cols_to_analyze = ["gender", "SeniorCitizen", "Partner", "Dependents", 
                   "PaymentMethod", "Contract"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_analyze):
    churn_rate = (
        df.groupby(col)["Churn"]
        .value_counts(normalize=True)
        .rename("Proportion")
        .mul(100)
        .reset_index()
    )
    
    sns.barplot(data=churn_rate, x=col, y="Proportion", hue="Churn", palette="pastel", ax=axes[i])
    axes[i].set_title(f"Churn by {col}")
    axes[i].set_ylabel("% Customers")
    axes[i].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.show()


# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt

# List of categorical variables to analyze
cols_to_analyze = ["gender", "senior_citizen", "partner", "dependents", 
                   "payment_method", "contract"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_analyze):
    churn_rate = (
        df.groupby(col)["churn"]
        .value_counts(normalize=True)
        .rename("Proportion")
        .mul(100)
        .reset_index()
    )
    
    sns.barplot(data=churn_rate, x=col, y="Proportion", hue="churn", palette="pastel", ax=axes[i])
    axes[i].set_title(f"Churn by {col}")
    axes[i].set_ylabel("% Customers")
    axes[i].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.show()


# In[68]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Group by tenure (month) to compute churn rate
churn_over_tenure = df.groupby("tenure")["churn"].apply(lambda x: (x.eq("Yes").mean() * 100)).reset_index()

# Rolling average (to smooth fluctuations)
churn_over_tenure["churn_Rolling"] = churn_over_tenure["churn"].rolling(window=6, center=True).mean()

# Plot churn rate over tenure
plt.figure(figsize=(10,6))
sns.lineplot(data=churn_over_tenure, x="tenure", y="churn", label="Monthly churn Rate", color="orange")
sns.lineplot(data=churn_over_tenure, x="tenure", y="churn_Rolling", label="6-Month Rolling Avg", color="blue")
plt.title("churn Trends Over Customer Lifecycle (Tenure in Months)")
plt.xlabel("Tenure (Months)")
plt.ylabel("churn Rate (%)")
plt.legend()
plt.show()


# In[ ]:




