####################################################################
# ---------------- Model blocks for building Transformers ----------------
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
        input_vocab_size,
        maximum_position_encoding,
        memory_length,
        segment_length,
        dropout_rate=0.1
    ):
        super(TransformerXLEncoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.memory_length = tf.constant(memory_length)
        self.segment_length = tf.constant(segment_length)

        # Input Embedding Layer
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size,
            d_model
        )
        # Sinusoidal positional encoding function
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        # Transformer XL Encoder Layer
        self.enc_layers = [
            XLEncoderLayer(
                d_model, 
                num_heads, 
                dff, 
                dropout_rate, 
                name = 'XLEncoderLayer-{}'.format(i + 1)
            ) for i in range(num_layers)
        ]


    def call(self, x):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        inputx = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        inputx *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputx = self.dropout(inputx, training=training)
    
        positional_encoding = self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            inputx = self.enc_layers[i](
                inputx, 
                positional_encoding,
                mask,
                training
            )

        return inputx  # (batch_size, input_seq_len, d_model)
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
        memory_length,
        segment_length,
        rate=0.1
    ):
        super(TransformerXLDecoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.memory_length = tf.constant(memory_length)
        self.segment_length = tf.constant(segment_length)
        
        # Ouput Embedding Layer
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            d_model
        )
        # Sinusoidal positional encoding function
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        # Transformer XL Decoder Layer
        self.dec_layers = [
            XLDecoderLayer(
                d_model,
                num_heads,
                dff,
                dropout_rate,
                name = 'XLDecoderLayer-{}'.format(i + 1)
            ) for i in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, x, look_ahead_mask, padding_mask, training=True):
        seq_len = tf.shape(x)[1]

        outputx = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        outputx *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputx = self.dropout(inputx, training=training)
    
        positional_encoding = self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            outputx, attention_weights = self.dec_layers[i](
                inputx, 
                positional_encoding,
                mask,
                training
            )

        return outputx, attention_weights  # (batch_size, output_seq_len, d_model)
## -------------------------------------------------------------------