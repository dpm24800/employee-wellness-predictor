## Pandas
```py
import numpy as np
import pandas as pd

# 2. LOAD DATA
df = pd.read_csv("cleanDataset.csv")
df.info()

print("Data Shape:", df.shape)
display(df.sample(5))


# df.isnull().sum()
# (df.isnull().sum() / len(df)) * 100
null_summary = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
})

print(null_summary)

# Numerical and Categorical Features
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print(categorical_cols.tolist())
print(numerical_cols.tolist())


for cat_col in categorical_cols:
    if cat_col == "flight":
        pass
    else:
        print(f"No of {cat_col}-s: {df[cat_col].nunique()},     {df[cat_col].unique().tolist()}")


# Dropping Columns
df = df.drop(columns=["City"])

df_clean_1['Churned'].value_counts(normalize=True)

# 1. Calculate correlation for all numeric columns
corr_matrix = df.corr(numeric_only=True)
# 2. Isolate the correlation with the target 'churned'
# We sort the values to make the heatmap easier to read
churn_corr = corr_matrix[['Churned']].sort_values(by='Churned', ascending=False)


encoder = OneHotEncoder(sparse_output=False, drop='first')
# Fit and transform each column individually
encoded = encoder.fit_transform(df_clean_1[[col]])
 # Convert to DataFrame with proper column names
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))

```