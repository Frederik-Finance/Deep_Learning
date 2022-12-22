import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

model = LinearRegression()
n_features_to_keep = 5

df = pd.read_excel('/root/deep_learning/data/datasets/Features.xlsx')

y_train = df['Close_DOGE']

x_train = df.drop(columns=['Close_DOGE'])

scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)

rfe = RFE(model, step=n_features_to_keep)

rfe.fit(x_train_scaled, y_train)

selected_features = x_train.columns[rfe.support_]
ranking = rfe.ranking_


x_train_selected = x_train_scaled[:, ranking <= n_features_to_keep]

ranking = rfe.ranking_

importance = rfe.estimator_.coef_

for feature, score in zip(x_train_selected, importance):
    print(f"{feature}: {score}")

importance = importance / importance.max()

sorted_idx = importance.argsort()

plt.barh(range(len(importance)), importance[sorted_idx], align='center')
plt.yticks(range(len(importance)), x_train_selected[sorted_idx])
plt.title('Feature Importance')
plt.show()
