import pandas as pd
import numpy as np
import preprocess
import models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

min_count = 2

df = pd.read_csv('train.csv')
auth2class = {'EAP': 0, 'HPL': 1, 'MWS': 2}
y = np.array([auth2class[a] for a in df.author])
y = to_categorical(y)


docs = preprocess.concatenate_texts(df.text)

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256
input_dim = max(max(i) for i in docs)
embedding_dims = 20

docs = pad_sequences(sequences=docs, maxlen=maxlen)


epochs = 24
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.2)

model = models.embed_model(input_dim, embedding_dims, 3)
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs)