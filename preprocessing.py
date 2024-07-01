import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('moscow_flats_dataset_eng.csv')

# Removing useless columns because of irrelevant or wrong data.
# is_apartments - describes whether a place has been registered as an apartment or not
# is_new - depends on the year, but has conflicting data - for 2016, some apartments are new and some not
# link - URL link to the source, not relevant for analysis.
df = df.drop(columns=['link', 'is_apartments', 'is_new'])

df = df.dropna()

# one hot encoding the regions of moscow, they are categorical data but cannot be divided
# numerically [Like CAR = 1, NWAR = 2 etc] because there is no proof of a relationship like that
# every region has expensive and cheap apartments
one_hot = pd.get_dummies(df['region_of_moscow'], dtype=float)
df = df.drop('region_of_moscow', axis=1)
df = df.join(one_hot)

X = df.drop(['price'], axis=1)
y = df['price']
# default train test split to check for p-values, 0.3 gave the best results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = sm.OLS(y_train, X_train).fit()

fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained")

# finding the right split
X = df.drop(['price'], axis=1)
y = df['price']

deviation_of_mse = []
test_size = []
deviation_of_training_mse = []

# run simple linear regression 50 times for each train test split and
# map the standard deviation of the MSE on train and test data
for x in range(10):
    t = 0.1 + x * 0.05
    mse_for_t = []
    mse_for_training = []
    for k in range(50):
        # do the same train test split 10 times with random selections
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t,
                                                            random_state=np.random.randint(1, 100))
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        mse_for_t.append(mean_squared_error(y_test, y_pred) / 1000000000000000)

        # for training data
        y_pred = reg.predict(X_train)
        mse_for_training.append(mean_squared_error(y_train, y_pred) / 1000000000000000)

    test_size.append(t)
    # find the std deviation of the test MSE and train MSE
    deviation_of_mse.append(np.std(mse_for_t))
    deviation_of_training_mse.append(np.std(mse_for_training))

fig.suptitle('Test size and relation to std deviation of MSE for Training and Test')

ax1.scatter(test_size, deviation_of_mse)
ax1.plot(test_size, deviation_of_mse)
ax1.set_xlabel('Test Size')
ax1.set_ylabel('Std deviation of Test MSE')

diff_in_mse = []

for x in range(10):
    diff_in_mse.append(deviation_of_mse[x] - deviation_of_training_mse[x])

ax2.scatter(test_size, deviation_of_training_mse)
ax2.plot(test_size, deviation_of_training_mse)
ax2.set_xlabel('Test Size')
ax2.set_ylabel('Std deviation of Train MSE')
# 0.4 seems ideal... but also try with 0.25, 3, 35

plt.show()
