import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetic_data.csv')

print("tổng quan inf")
print(df.info())
print("\nChi tiết")
print(df.describe())


# chọn cột liên quan
cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'A1Cresult', 'insulin', 'change', 'diabetesMed'
]

#hít

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (18, 20)

#histo
df[cols].hist(bins=20, color='skyblue', edgecolor='black', layout=(4, 3))
plt.suptitle("Biểu đồ Hist", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#check mis dât
df_nan = df[cols].copy()
df_nan.replace('?', np.nan, inplace=True)
print("Missing values:")
print(df_nan.isna().sum())


print("A1C tk:")
print(df['A1Cresult'].value_counts())
print("\nInsulin tk:")
print(df['insulin'].value_counts())


# --> mapping A1Cresult, insulin
mapping_a1c = {'None': 0, 'Norm': 5, '>7': 7, '>8': 8}
df_nan['A1Cresult'] = df_nan['A1Cresult'].map(mapping_a1c).fillna(0)
mapping_med = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
df_nan['insulin'] = df_nan['insulin'].map(mapping_med).fillna(0)
df_nan['change'] = df_nan['change'].apply(lambda x: 1 if x == 'Ch' else 0)
df_nan['diabetesMed'] = df_nan['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0)
df_clustering = df_nan.fillna(df_nan.median())
# A1Result bị missing --> mã hóa tesxt -> số --> fill median

#Ktra outliers
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig.suptitle('Biểu đồ Box Plot cho từng đặc trưng', fontsize=20)
for i, col in enumerate(df_nan):
    row = i // 3
    axis = i % 3
    sns.boxplot(x=df_nan[col], ax=axes[row, axis], color='green', fliersize=5)
    axes[row, axis].set_title(f'Phân phối của {col}', fontsize=14)
    axes[row, axis].set_xlabel('')
plt.show()
#không loại outliers vì những số không bth có thể sẽ ah đến chẩn đoán
df_sample = df_clustering.sample(n=1000, random_state=42)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_sample)

df_sample.to_csv('processed_diabetes_1000.csv', index=False)