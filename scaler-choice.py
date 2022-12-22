import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler

df = pd.read_excel('/root/deep_learning/data/datasets/Features.xlsx')

X = df.drop("Close_DOGE", axis=1)
y = df["Close_DOGE"]

mean = X.mean()
std = X.std()

iqr = X.quantile(0.75) - X.quantile(0.25)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skewness = pd.DataFrame(X_scaled).skew()
kurtosis = pd.DataFrame(X_scaled).kurtosis()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

range_skewness = pd.DataFrame(X_scaled).skew()
range_kurtosis = pd.DataFrame(X_scaled).kurtosis()

scaler = PowerTransformer(method="yeo-johnson")
X_scaled = scaler.fit_transform(X)

transformed_skewness = pd.DataFrame(X_scaled).skew()
transformed_kurtosis = pd.DataFrame(X_scaled).kurtosis()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

robust_skewness = pd.DataFrame(X_scaled).skew()
robust_kurtosis = pd.DataFrame(X_scaled).kurtosis()

print("StandardScaler:")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")
print()
print("MinMaxScaler:")
print(f"Skewness: {range_skewness}")
print(f"Kurtosis: {range_kurtosis}")
print()
print("PowerTransformer:")
print(f"Skewness: {transformed_skewness}")
print(f"Kurtosis: {transformed_kurtosis}")
print()
print("RobustScaler:")
print(f"Skewness: {robust_skewness}")
print(f"Kurtosis: {robust_kurtosis}")
