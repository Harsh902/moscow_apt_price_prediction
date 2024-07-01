import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import PredictionErrorDisplay
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)

X_train = pd.read_csv('Dataframes/shifted_training_X.csv')
y_train = pd.read_csv('Dataframes/shifted_training_Y.csv')
X_val = pd.read_csv('Dataframes/shifted_validation_X.csv')
y_val = pd.read_csv('Dataframes/shifted_validation_Y.csv')

# plt.scatter(np.log(X_train['total_area']), np.sqrt(np.log(y_train)))
# plt.xlabel('log(total_area)')
# plt.ylabel('sqrt(np.log(total_area))')
# plt.show()

# building simple model with total area and price
# but applying our transformations
total_area_log = X_train['total_area'].apply(np.log)
total_area_log = total_area_log.to_frame()
total_area_combined = total_area_log.join(X_train['total_area'], lsuffix='_log', rsuffix='_normal')

# joining with construction year
total_area_combined = total_area_combined.join(X_train['construction_year'])
# # adding CAR stuff
total_area_combined = total_area_combined.join(X_train['CAR'])


price_sqrt_log = y_train.apply(np.log).apply(np.sqrt)

model = LinearRegression()
model.fit(total_area_combined, price_sqrt_log)

# X_val and y_val
area_val = X_val['total_area'].apply(np.log)
area_val = area_val.to_frame()
area_val = area_val.join(X_val['total_area'], lsuffix='_log', rsuffix='_normal')

area_val = area_val.join(X_val['construction_year'])
area_val = area_val.join(X_val['CAR'])

price_val = y_val.apply(np.log).apply(np.sqrt)
y_pred = model.predict(area_val)
mse = mean_squared_error(price_val, y_pred)
print(mse)

display = PredictionErrorDisplay(y_true=price_val, y_pred=y_pred)
display.plot()
plt.show()

print(model.score(area_val, price_val))

with open('models/car_year_area_price.pkl', "wb") as f:
    pickle.dump(model, f)

# print(y_train.join(X_train).corr())
