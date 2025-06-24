import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(input_dim: int, latent_dim: int = 2):
    inp = layers.Input(shape=(input_dim,))
    x   = layers.Dense(8, activation='relu')(inp)
    z   = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    x   = layers.Dense(8, activation='relu')(z)
    out = layers.Dense(input_dim, activation='linear')(x)
    return Model(inp, out, name='AE')

def train_autoencoder(train_df, val_df, latent_dim=2, epochs=100, batch=32):
    ae = build_autoencoder(train_df.shape[1], latent_dim)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(train_df, train_df,
           validation_data=(val_df, val_df),
           epochs=epochs, batch_size=batch, verbose=0)
    # Extract encoder
    encoder = Model(ae.input, ae.get_layer('latent').output)
    return encoder 