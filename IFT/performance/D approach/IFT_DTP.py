import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from sklearn.model_selection import train_test_split

df = pd.read_excel('../database/data.xlsx')

df = df[['P(Mpa)', 'T (R )', 'delta density(g/cm3)', 'Tc®', 'Pc', 'IFT(mN/m)']]
df[df['P(Mpa)'] == 'sat']=np.nan
df = df.dropna(axis=0,how='any')
print(df.shape[0])
print(df.head())

X = df.values[:,:5].astype(float)
y = df.values[:,5]

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
print("mean =", mean, "std =", std)

n = df.shape[0]

model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", metrics=["mae"])
history = model.fit(X, y, validation_data=(X,y), epochs=2000, batch_size=128)
test_predictions = model.predict(X)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

#save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

from sklearn.metrics import r2_score
r_squared = r2_score(y, test_predictions, multioutput='raw_values')

import matplotlib.pyplot as plt
plt.scatter(y, test_predictions)
plt.xlabel('True Values [Hydrocarbon-water interfacial tension]')
plt.ylabel('Predictions [Hydrocarbon-water interfacial tension]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 200], [-100, 200])
plt.text(40, 30, 'R-squared = %0.2f' % r_squared)
plt.show()

plt.clf()

#save r-square to csv: 
column1 = y
column2 = test_predictions.reshape(test_predictions.shape[0],)
rsquare = {'test data':column1.tolist(), 'predict':column2.tolist()}
rsquare_df = pd.DataFrame(rsquare)
rsquare_df.to_csv('Rsquare.csv')

history_dict = history.history
print(history_dict.keys())

loss_values = history_dict['loss']
mean_absolute_error = history_dict['mean_absolute_error']
val_mean_absolute_error = history_dict['val_mean_absolute_error']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, mean_absolute_error, 'bo', label='Training mean_absolute_error')
plt.plot(epochs, val_mean_absolute_error, 'b', label='val_mean_absolute_error')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()