from keras.models import Sequential, Model
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input, multiply, Conv1D, MaxPooling1D, \
    Flatten, Dropout, LSTM, Bidirectional


def embed_model(input_dim, embedding_dims, n_classes, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def merged_model(input_dim, maxlen, n_classes, embed_dims=(10, 20, 30, 100), densify=False, optimizer='adam'):
    # Probably want to change parameters and activations here or number of layers
    i1 = Input(shape=(maxlen,))
    e1 = Embedding(input_dim=input_dim, output_dim=embed_dims[0])(i1)
    ga1 = GlobalAveragePooling1D()(e1)
    d1 = Dense(n_classes, activation="softmax")(ga1)

    i2 = Input(shape=(maxlen,))
    e2 = Embedding(input_dim=input_dim, output_dim=embed_dims[1])(i2)
    ga2 = GlobalAveragePooling1D()(e2)
    d2 = Dense(n_classes, activation="softmax")(ga2)

    i3 = Input(shape=(maxlen,))
    e3 = Embedding(input_dim=input_dim, output_dim=embed_dims[2])(i3)
    ga3 = GlobalAveragePooling1D()(e3)
    d3 = Dense(n_classes, activation="softmax")(ga3)

    i4 = Input(shape=(maxlen,))
    e4 = Embedding(input_dim=input_dim, output_dim=embed_dims[3])(i4)
    ga4 = GlobalAveragePooling1D()(e4)
    d4 = Dense(n_classes, activation="softmax")(ga4)

    # Probably want to change layers here like average, concatenate, multiply etc
    # https://keras.io/layers/merge/
    merge1 = multiply([d1, d2, d3, d4])
    if densify:
        dfinal = Dense(n_classes, activation="softmax")(merge1)
        model = Model(inputs=[i1, i2, i3, i4], outputs=dfinal)
    else:
        model = Model(inputs=[i1, i2, i3, i4], outputs=merge1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def char_cnn_model(input_shape, n_classes, optimizer="adam"):
    model = Sequential()
    model.add(Conv1D(256, 7, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(256, 7, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def nn_conv_maxpoool_bilstm_dropout(vocab_size, embed_size, input_len, cnn_filters,
                                    cnn_kernel_size, pool_size, bilstm_size, drop_rate, n_classes, optimizer="adam"):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=input_len))
    model.add(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Bidirectional(LSTM(bilstm_size)))
    model.add(Dropout(drop_rate))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model