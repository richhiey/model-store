##############################################################################################################
# ---------------- Model blocks for building Transformers ----------------
##############################################################################################################
import tensorflow as tf
from .utils import positional_encoding
from .layers import XLEncoderLayer, \
                    XLDecoderLayer, \
                    Memory
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


    def call(self, inputs, positional_encoding, training, padding_mask=None):
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


    def call(self, inputs, encoder_outputs, positional_encoding, training, padding_mask=None):
        ## -------------------------------------------------------------------
        inputs, memory_length = inputs
        seq_len = tf.shape(inputs)[1]
        ## -------------------------------------------------------------------
        embeddings = self.embedding(inputs)                                 # (batch_size, input_seq_len, d_model)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.dropout(embeddings, training=training)
        ## -------------------------------------------------------------------
        positional_encoding = positional_encoding[:, :2*seq_len, :]
        ## -------------------------------------------------------------------
        for i in range(self.num_layers):
            outputx, attention_weights = self.dec_layers[i](
                embeddings,
                encoder_outputs,
                positional_encoding,
                memory_length,
                padding_mask,
                training
            )
        ## -------------------------------------------------------------------
        return outputx, attention_weights                                   # (batch_size, output_seq_len, d_model)
        ## -------------------------------------------------------------------
##############################################################################################################
