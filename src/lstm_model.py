from tensorflow.keras import layers, Model
import tensorflow as tf

def build_lstm(window, latent_dim, hidden=32, dropout=0.0, l2_reg=0.0):
    inp = layers.Input(shape=(window, latent_dim))
    
    # Add regularization if specified
    if l2_reg > 0:
        regularizer = tf.keras.regularizers.l2(l2_reg)
    else:
        regularizer = None
    
    # LSTM with dropout and regularization
    if dropout > 0:
        x = layers.LSTM(hidden, 
                       dropout=dropout,
                       kernel_regularizer=regularizer)(inp)
    else:
        x = layers.LSTM(hidden, 
                       kernel_regularizer=regularizer)(inp)
    
    out = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inp, out, name='AE_LSTM')
    return model

def train_lstm(X_train, y_train, X_val, y_val,
               window, latent_dim, epochs=50, batch=64, 
               dropout=0.0, l2_reg=0.0):
    lstm = build_lstm(window, latent_dim, dropout=dropout, l2_reg=l2_reg)
    lstm.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['AUC'])
    lstm.fit(X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=epochs, batch_size=batch, verbose=0)
    return lstm 