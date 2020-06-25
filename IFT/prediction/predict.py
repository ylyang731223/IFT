import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping

from keras.models import Model, load_model

df = pd.read_excel('hexane.xlsx')
#df = pd.read_excel('tridecene.xlsx')
#df = pd.read_excel('benzene.xlsx')
#df = pd.read_excel('toluene.xlsx')
#df = pd.read_excel('butanol.xlsx')
#df = pd.read_excel('dodecanol.xlsx')
print(df.head())

max_length = 17
print("max_length",max_length)

word_index = {'C': 1, 'c': 2, '1': 3, 'O': 4, '(': 5, ')': 6, '=': 7}
print("word_index =", word_index)

X_SMILES = df.values[:,0]
X = np.zeros((X_SMILES.shape[0],max_length), dtype=int)
for i in range(X_SMILES.shape[0]):
	for j in range(len(X_SMILES[i])):
		k = max_length-len(X_SMILES[i])
		X[i][j+k] = word_index[X_SMILES[i][j]]

X_df = pd.DataFrame(X)
X_df.to_excel("X_pred.xlsx")

df = df[['P', 'T']]
print(df.head())
numerical_data_x = df.values[:,:2].astype(float)
numerical_data_x = np.array(numerical_data_x)

mean = np.array([17.43087025, 581.8341254])
std = np.array([41.14258117, 74.39028753])

numerical_data_x = (numerical_data_x - mean) / std

X = np.append(X, numerical_data_x, axis=1)
print("X shape =", X.shape)

model = load_model('my_model.h5')

tension = model.predict({'nlp_input': X[:,:max_length], 'meta_input': X[:,max_length:]})

Ten_df = pd.DataFrame(tension)
Ten_df.to_excel("Ten_pred.xlsx")
print(tension)
