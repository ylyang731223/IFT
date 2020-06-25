import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

df = pd.read_excel('../database/data.xlsx')

df['smiles'] = ''

print(df.columns)
df = df.drop(columns=['no', 'year', 'IFT ref', 'Tr', 'Pr'])
df[df['P(Mpa)'] == 'sat']=np.nan
df = df.dropna(axis=0,how='any')

for i in range(df.shape[0]):
    if df.iloc[i,0] == 'methane':
        df.iloc[i,7] = 'C'
    elif df.iloc[i,0] == 'propane':
        df.iloc[i,7] = 'CCC'
    elif df.iloc[i,0] == 'butane':
        df.iloc[i,7] = 'CCCC'
    elif df.iloc[i,0] == 'pentane':
        df.iloc[i,7] = 'CCCCC'
    elif df.iloc[i,0] == 'hexane':
        df.iloc[i,7] = 'CCCCCC'
    elif df.iloc[i,0] == 'heptane':
        df.iloc[i,7] = 'CCCCCCC'
    elif df.iloc[i,0] == 'octane':
        df.iloc[i,7] = 'CCCCCCCC'
    elif df.iloc[i,0] == 'nonane':
        df.iloc[i,7] = 'CCCCCCCCC'
    elif df.iloc[i,0] == 'decane':
        df.iloc[i,7] = 'CCCCCCCCCC'
    elif df.iloc[i,0] == 'undecane':
        df.iloc[i,7] = 'CCCCCCCCCCC'
    elif df.iloc[i,0] == 'dodecane':
        df.iloc[i,7] = 'CCCCCCCCCCCC'
    elif df.iloc[i,0] == 'tridecane':
        df.iloc[i,7] = 'CCCCCCCCCCCCC'
    elif df.iloc[i,0] == 'tetradecane':
        df.iloc[i,7] = 'CCCCCCCCCCCCCC'
    elif df.iloc[i,0] == 'hexadecane':
        df.iloc[i,7] = 'CCCCCCCCCCCCCCCC'
    elif df.iloc[i,0] == 'benzene':
        df.iloc[i,7] = 'c1ccccc1'
    elif df.iloc[i,0] == 'toluene':
        df.iloc[i,7] = 'Cc1ccccc1'
    elif df.iloc[i,0] == 'Toluene':
        df.iloc[i,7] = 'Cc1ccccc1'
    elif df.iloc[i,0] == 'isooctane':
        df.iloc[i,7] = 'CC(C)CC(C)(C)C'
    elif df.iloc[i,0] == 'cyclohexane':
        df.iloc[i,7] = 'C1CCCCC1'
    elif df.iloc[i,0] == 'pxylene':
        df.iloc[i,7] = 'CC1=CC=C(C)C=C1'
    elif df.iloc[i,0] == 'mxylene':
        df.iloc[i,7] = 'CC1=CC(C)=CC=C1'
    elif df.iloc[i,0] == 'oxylene':
        df.iloc[i,7] = 'CC1=C(C)C=CC=C1'
    elif df.iloc[i,0] == 'isohexane':
        df.iloc[i,7] = 'CCCC(C)C'
    elif df.iloc[i,0] == 'isopentane':
        df.iloc[i,7] = 'CCC(C)C'
    elif df.iloc[i,0] == 'isopropylbenzene':
        df.iloc[i,7] = 'CC(C)c1ccccc1'
    elif df.iloc[i,0] == 'methylcyclohexane-water':
        df.iloc[i,7] = 'CC1CCCCC1'
    elif df.iloc[i,0] == '3-methylpentane-water':
        df.iloc[i,7] = 'CCC(C)CC'
    elif df.iloc[i,0] == '2,3-dimethylpentane-water':
        df.iloc[i,7] = 'CCC(C)C(C)C'
    elif df.iloc[i,0] == 'Propylbenzene':
        df.iloc[i,7] = 'CCCC1=CC=CC=C1'
    elif df.iloc[i,0] == 'ethylbenzene':
        df.iloc[i,7] = 'CCc1ccccc1'
    elif df.iloc[i,0] == 'Ethylbenzene':
        df.iloc[i,7] = 'CCc1ccccc1'
    elif df.iloc[i,0] == 'Butylbenzene':
        df.iloc[i,7] = 'CCCCC1=CC=CC=C1'
    elif df.iloc[i,0] == 'butanol':
        df.iloc[i,7] = 'CCCCO'
    elif df.iloc[i,0] == 'pentanol':
        df.iloc[i,7] = 'CCCCCO'
    elif df.iloc[i,0] == 'hexanol':
        df.iloc[i,7] = 'CCCCCCO'
    elif df.iloc[i,0] == 'heptanol':
        df.iloc[i,7] = 'CCCCCCCO'
    elif df.iloc[i,0] == 'octanol':
        df.iloc[i,7] = 'CCCCCCCCO'
    elif df.iloc[i,0] == 'nonanol':
        df.iloc[i,7] = 'CCCCCCCCCO'
    elif df.iloc[i,0] == 'decanol':
        df.iloc[i,7] = 'CCCCCCCCCCO'
    elif df.iloc[i,0] == 'dodecanol':
        df.iloc[i,7] = 'CCCCCCCCCCCCO'
    elif df.iloc[i,0] == 'Tridecene':
        df.iloc[i,7] = 'CCCCCCCCCCCC=C'
    elif df.iloc[i,0] == 'Tetradecene':
        df.iloc[i,7] = 'CCCCCCCCCCCCC=C'
    elif df.iloc[i,0] == 'Pentadecene':
        df.iloc[i,7] = 'CCCCCCCCCCCCCC=C'
    elif df.iloc[i,0] == 'Hexadecene':
        df.iloc[i,7] = 'CCCCCCCCCCCCCCC=C'

df = df.drop(columns=['components', 'delta density(g/cm3)'])
df = df[['smiles', 'P(Mpa)', 'T (R )', 'Tc®', 'IFT(mN/m)']]
print(df.head())

smile_data_x = df.values[:,0]
numerical_data_x = df.values[:,1:4].astype(float)
y = df.values[:,4]

mean = numerical_data_x.mean(axis=0)
std = numerical_data_x.std(axis=0)
numerical_data_x = (numerical_data_x - mean) / std
print("mean =", mean, "std =", std)

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# maximum length of sequence, everything afterwards is discarded!
max_length = max([len(df.values[i,0]) for i in range(df.shape[0])])
print("max_length",max_length)

#create and fit tokenizer
tokenizer = Tokenizer(lower=False, char_level=True)
tokenizer.fit_on_texts(df.values[:,0])

#represent input data as word rank number sequences
X = tokenizer.texts_to_sequences(df.values[:,0])
X = sequence.pad_sequences(X, dtype='int32', truncating='pre', value=0., maxlen=max_length)
X = np.append(X, numerical_data_x, axis=1)

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model

nlp_input = Input(shape=(max_length,), dtype='int32', name='nlp_input')

z = Embedding(output_dim=5, input_dim=len(tokenizer.word_index)+1, input_length=max_length)(nlp_input)

nlp_out = LSTM(32)(z)

meta_input = Input(shape=(3,), name='meta_input')
z = concatenate([nlp_out, meta_input])

z = Dense(16, activation='relu')(z)
z = Dense(16, activation='relu')(z)
z = Dense(16, activation='relu')(z)
main_output = Dense(1, name='main_output')(z)

model = Model(inputs=[nlp_input, meta_input], outputs=[main_output])
model.compile(optimizer='rmsprop',
              loss={'main_output': 'mse'},
              loss_weights={'main_output': 1.}, metrics=['mae'])
history = model.fit({'nlp_input': X[:,:max_length], 'meta_input': X[:,max_length:]},
          {'main_output': y}, validation_data=({'nlp_input': X[:,:max_length], 'meta_input': X[:,max_length:]},{'main_output': y}),
          epochs=2000, batch_size=32)
test_predictions = model.predict({'nlp_input': X[:,:max_length], 'meta_input': X[:,max_length:]})

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