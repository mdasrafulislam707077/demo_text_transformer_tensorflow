import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import numpy as np
import pickle


demo_sentences = [
      
]







tokenizer = Tokenizer()
tokenizer.fit_on_texts(demo_sentences)
sequences = tokenizer.texts_to_sequences(demo_sentences)

max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

X_train = padded_sequences
y_train = np.roll(padded_sequences, shift=-1, axis=-1)
y_train[:, -1] = 0  
sample_weights = np.where(y_train != 0, 1.0, 0.0)



def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pos_enc, dtype=tf.float32)

def transformer_block(inputs, num_heads, ff_dim, d_model, dropout_rate=0.1):
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model)(inputs, inputs, use_causal_mask=True)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_gpt_model(vocab_size, seq_len, d_model=128, num_heads=4, num_layers=4, ff_dim=512):
    inputs = tf.keras.Input(shape=(seq_len,))
    token_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)(inputs)
    pos_encoding = get_positional_encoding(seq_len, d_model)
    x = token_embeddings + pos_encoding
    
    for _ in range(num_layers):
        x = transformer_block(x, num_heads, ff_dim, d_model)
    
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)



vocab_size = len(tokenizer.word_index) + 1
seq_len = max_len

gpt_model = build_gpt_model(vocab_size, seq_len)
gpt_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
gpt_model.summary()

gpt_model.fit(X_train, y_train, sample_weight=sample_weights, epochs=7, batch_size=32)



gpt_model.save('gpt_model.keras')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved successfully!")