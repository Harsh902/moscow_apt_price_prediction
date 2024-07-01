import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('moscow_flats_cleaned.csv')

X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4,
                                                  random_state=42)  # 0.4 * 0. 7 = 0.3

# normalising the data based on training set.
# X_train, X_val and X_test
mean = X_train.mean()
std = X_train.std()
normalized_training_X = (X_train - mean) / std
normalized_validation_X = (X_val - mean) / std
normalized_test_X = (X_test - mean) / std

# y_train, y_val and y_test
mean = y_train.mean()
std = y_train.std()
normalized_training_Y = (y_train - mean) / std
normalized_validation_Y = (y_val - mean) / std
normalized_test_Y = (y_test - mean) / std


# shifting the data to avoid -ve values
# X_train
minimums = normalized_training_X.min()
shifted_X_train = (normalized_training_X - minimums) + 1
shifted_X_train.to_csv('Dataframes/shifted_training_X.csv', index=False)

# X_val
minimums = normalized_validation_X.min()
shifted_X_val = (normalized_validation_X - minimums) + 1
shifted_X_val.to_csv('Dataframes/shifted_validation_X.csv', index=False)

# X_test
minimums = normalized_test_X.min()
shifted_X_test = (normalized_test_X - minimums) + 1
shifted_X_test.to_csv('Dataframes/shifted_test_X.csv', index=False)

# y_train
minimums = normalized_training_Y.min()
shifted_y_train = (normalized_training_Y - minimums) + 1
shifted_y_train.to_csv('Dataframes/shifted_training_Y.csv', index=False)

# y_val
minimums = normalized_validation_Y.min()
shifted_y_val = (normalized_validation_Y - minimums) + 1
shifted_y_val.to_csv('Dataframes/shifted_validation_Y.csv', index=False)

# y_test
minimums = normalized_test_Y.min()
shifted_y_test = (normalized_test_X - minimums) + 1
shifted_y_test.to_csv('Dataframes/shifted_test_Y.csv', index=False)


plt.scatter(np.log(shifted_X_train['total_area']), np.sqrt(np.log(shifted_y_train)))
# plt.scatter(np.log(shifted_X_val['total_area']), np.sqrt(np.log(shifted_y_val)))
plt.xlabel('log(total_area)')
plt.ylabel('sqrt(np.log(total_area))')
plt.show()

