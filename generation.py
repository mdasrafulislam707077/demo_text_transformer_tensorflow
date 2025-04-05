import tensorflow as tf
import pickle
import numpy as np

class TextGenerator:
    def __init__(self, model_path='gpt_model.keras', tokenizer_path='tokenizer.pkl'):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'get_positional_encoding': self.get_positional_encoding,
                'transformer_block': self.transformer_block
            }
        )
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        self.max_len = self.model.input_shape[1]
    
    @staticmethod
    def get_positional_encoding(seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc = np.zeros((seq_len, d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pos_enc, dtype=tf.float32)
    
    @staticmethod
    def transformer_block(inputs, num_heads=4, ff_dim=512, d_model=128, dropout_rate=0.1):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model)(inputs, inputs, use_causal_mask=True)
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation='relu')(out1)
        ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    def sample_with_temperature(self, logits, temperature=1.0):
        if temperature <= 0:
            return np.argmax(logits)
        scaled_logits = logits / temperature
        probs = np.exp(scaled_logits - np.max(scaled_logits))
        probs /= np.sum(probs)
        return np.random.choice(len(probs), p=probs)
    
    def generate_text(self, seed_text, max_gen=20, temperature=0.7):
        generated = seed_text
        for _ in range(max_gen):
            tokens = self.tokenizer.texts_to_sequences([generated])[0]
            if len(tokens) >= self.max_len:
                tokens = tokens[-self.max_len:]
            
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [tokens], maxlen=self.max_len, padding='pre')
            
            probs = self.model.predict(padded, verbose=0)[0][-1]
            sampled_idx = self.sample_with_temperature(probs, temperature)
            print(self.tokenizer.index_word.get(sampled_idx, ''))
            if self.tokenizer.index_word.get(sampled_idx, '') == "endchatdeathline":
                  break
            if sampled_idx == 0:
                continue

            generated += " " + self.tokenizer.index_word.get(sampled_idx, '')
        
        return generated

if __name__ == "__main__":
    generator = TextGenerator()
    
    while True:
        seed = input("\nEnter seed text (or 'quit' to exit): ")
        if seed.lower() == 'quit':
            break
        
        temperature = .1
        length = 10
        
        result = generator.generate_text(seed, max_gen=length, temperature=temperature)
        # if result.endswith(("end_chat")):
        #         break
        # else:
        print("\nGenerated text:")
        print(result)