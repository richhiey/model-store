####################################################################
# ---------------- Blocks for building Transformers ----------------
####################################################################
import tensorflow as tf
from .utils import positional_encoding
from .layers import EncoderLayer, DecoderLayer,
                    XLEncoderLayer, XLDecoderLayer,
                    Memory
####################################################################

## -------------------------------------------------------------------
class TransformerEncoder(tf.keras.Model):
    def __init__(
        self, 
        num_layers, 
        d_model, 
        num_heads, 
        dff, 
        input_vocab_size,
        maximum_position_encoding, 
        rate = 0.1
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, 
            d_model
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, 
            d_model
        )
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
   
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # adding embedding and position encoding.
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
    
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
    
        return x  # (batch_size, input_seq_len, d_model)
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class TransformerDecoder(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1
    ):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            d_model
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, look_ahead_mask, padding_mask, training=True):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
    
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
    
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](
                x, training, look_ahead_mask, padding_mask
            )
      
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
    
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class TransformerXLEncoder(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1
    ):
        super(TransformerXLEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            d_model
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        self.memories = [
            Memory(batch_size, memory_len, segment_len, d_model) for _ in range(num_layers)
        ]
        self.enc_layers = [
            XLEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x):
        pass
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class TransformerXLDecoder(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1
    ):
        super(TransformerXLDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            d_model
        )
        batch_size = 32
        segment_len = 512
        memory_len = segment_len
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        self.memories = [
            Memory(batch_size, memory_len, segment_len, d_model) for _ in range(num_layers)
        ]
        self.dec_layers = [
            XLDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x):
        pass
## -------------------------------------------------------------------