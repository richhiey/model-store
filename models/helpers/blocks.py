##############################################################################################################
# ---------------- Model blocks for building Transformers ----------------
##############################################################################################################
import tensorflow as tf
from .utils import positional_encoding
from .layers import EncoderLayer, \
                    DecoderLayer, \
                    XLEncoderLayer, \
                    XLDecoderLayer, \
                    Memory
##############################################################################################################


##############################################################################################################
## -------------------------------------------------------------------
## TENSORFLOW MODELS FOR VANILLA TRANSFORMER
## -------------------------------------------------------------------
## Transformer paper - https://arxiv.org/abs/1706.03762
## -------------------------------------------------------------------
##############################################################################################################
class TransformerEncoderStack(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                    maximum_position_encoding, rate = 0.1):
        ## -------------------------------------------------------------------
        super(TransformerEncoder, self).__init__()
        ## -------------------------------------------------------------------
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        ## -------------------------------------------------------------------
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size,
            d_model
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, 
            d_model
        )
        self.dropout = tf.keras.layers.Dropout(rate)
        ## -------------------------------------------------------------------
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        ## -------------------------------------------------------------------

   
    def call(self, x, training, mask):
        ## -------------------------------------------------------------------
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)                                               # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ## -------------------------------------------------------------------
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        ## -------------------------------------------------------------------
        return x                                                           # (batch_size, input_seq_len, d_model)
        ## -------------------------------------------------------------------
##############################################################################################################


##############################################################################################################
class TransformerDecoderStack(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                    maximum_position_encoding, rate=0.1):
        ## -------------------------------------------------------------------
        super(TransformerDecoder, self).__init__()
        ## -------------------------------------------------------------------
        self.num_layers = num_layers
        self.d_model = d_model
        ## -------------------------------------------------------------------
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size,
            d_model
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model
        )
        self.dropout = tf.keras.layers.Dropout(rate)
        ## -------------------------------------------------------------------
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        ## -------------------------------------------------------------------


    def call(self, x, look_ahead_mask, padding_mask, training=True):
        ## -------------------------------------------------------------------
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        ## -------------------------------------------------------------------
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        ## -------------------------------------------------------------------
        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i](
                x, training, look_ahead_mask, padding_mask
            )
        ## -------------------------------------------------------------------
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        ## -------------------------------------------------------------------    
        return x, attention_weights
        ## -------------------------------------------------------------------
##############################################################################################################


##############################################################################################################
class TransformerXLEncoderStack(tf.keras.Model):

    def __init__(self, configs, **kwargs):
        ## -------------------------------------------------------------------
        super(TransformerXLEncoderStack, self).__init__(**kwargs)
        ## -------------------------------------------------------------------
        self.num_layers = int(configs['num_layers'])
        self.d_model = int(configs['d_model'])
        self.num_heads = int(configs['num_heads'])
        self.dff = int(configs['dff'])
        self.maximum_position_encoding = int(configs['maximum_position_encoding'])
        self.memory_length = tf.constant(int(configs['memory_length']))
        self.max_sequence_length = tf.constant(int(configs['max_sequence_length']))
        self.dropout_rate = float(configs['dropout_rate'])
        ## -------------------------------------------------------------------
        self.enc_layers = [
            XLEncoderLayer(
                self.d_model, 
                self.num_heads, 
                self.dff, 
                self.memory_length,
                self.max_sequence_length, 
                self.dropout_rate,
                name = 'XLEncoderLayer-{}'.format(i + 1)
            ) for i in tf.range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        ## -------------------------------------------------------------------


    def call(self, inputs, positional_encoding, padding_mask=None, training=True):
        ## -------------------------------------------------------------------
        embeddings, memory_length = inputs
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.dropout(embeddings, training=training)
        seq_len = tf.shape(embeddings)[1]
        ## -------------------------------------------------------------------
        positional_encoding = positional_encoding[:, :2*seq_len, :]
        ## -------------------------------------------------------------------
        for i in range(self.num_layers):
            embeddings = self.enc_layers[i](
                embeddings, 
                positional_encoding,
                memory_length,
                padding_mask,
                training,
            )
        ## -------------------------------------------------------------------        
        return embeddings                                                     # (batch_size, input_seq_len, d_model)
        ## -------------------------------------------------------------------
##############################################################################################################


##############################################################################################################
class TransformerXLDecoderStack(tf.keras.Model):

    def __init__(self, configs, **kwargs):
        ## -------------------------------------------------------------------
        super(TransformerXLDecoderStack, self).__init__(**kwargs)
        ## -------------------------------------------------------------------
        self.num_layers = int(configs['num_layers'])
        self.d_model = int(configs['d_model'])
        self.num_heads = int(configs['num_heads'])
        self.dff = int(configs['dff'])
        self.target_vocab_size = int(configs['target_vocab_size'])
        self.maximum_position_encoding = int(configs['maximum_position_encoding'])
        self.memory_length = tf.constant(int(configs['memory_length']))
        self.dropout_rate = float(configs['dropout_rate'])
        self.max_sequence_length = tf.constant(int(configs['max_sequence_length']))
        ## -------------------------------------------------------------------
        self.embedding = tf.keras.layers.Embedding(
            self.target_vocab_size,
            self.d_model
        )
        ## -------------------------------------------------------------------
        self.dec_layers = [
            XLDecoderLayer(
                self.d_model, 
                self.num_heads, 
                self.dff, 
                self.memory_length,
                self.max_sequence_length, 
                self.dropout_rate,
                name = 'XLEncoderLayer-{}'.format(i + 1)
            ) for i in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        ## -------------------------------------------------------------------


    def call(self, x, encoder_outputs, positional_encoding, look_ahead_mask, padding_mask=None, training=True):
        ## -------------------------------------------------------------------
        x, memory_length = x
        seq_len = tf.shape(x)[1]
        ## -------------------------------------------------------------------
        outputx = self.embedding(x)                                         # (batch_size, input_seq_len, d_model)
        outputx *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        outputx = self.dropout(outputx, training=training)
        ## -------------------------------------------------------------------
        positional_encoding = positional_encoding[:, :2*seq_len, :]
        ## -------------------------------------------------------------------
        for i in range(self.num_layers):
            outputx, attention_weights = self.dec_layers[i](
                outputx,
                encoder_outputs,
                positional_encoding,
                memory_length,
                look_ahead_mask,
                padding_mask,
                training
            )
        ## -------------------------------------------------------------------
        return outputx, attention_weights                                   # (batch_size, output_seq_len, d_model)
        ## -------------------------------------------------------------------
##############################################################################################################
