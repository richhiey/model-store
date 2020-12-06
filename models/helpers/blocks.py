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

    def __init__(self, num_layers, d_model, num_heads, dff, memory_length,
        max_sequence_length, **kwargs):
        ## -------------------------------------------------------------------
        super(TransformerXLEncoderStack, self).__init__(**kwargs)
        ## -------------------------------------------------------------------
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.memory_length = tf.constant(memory_length)
        self.max_sequence_length = tf.constant(max_sequence_length)
        self.dropout_rate = 0.1
        ## -------------------------------------------------------------------
        self.enc_layers = [
            XLEncoderLayer(
                self.d_model, 
                self.num_heads, 
                self.dff, 
                self.memory_length,
                self.max_sequence_length, 
                rate = self.dropout_rate,
                layer_id = i + 1,
                name = 'XLEncoderLayer-{}'.format(i + 1)
            ) for i in tf.range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        ## -------------------------------------------------------------------


    def call(self, embeddings, memory_length, positional_encoding, 
        t_mask, padding_mask, training):
        ## -------------------------------------------------------------------
        for i in range(self.num_layers):
            embeddings = self.enc_layers[i](
                embeddings,
                memory_length,
                positional_encoding,
                t_mask,
                padding_mask,
                training
            )
        ## -------------------------------------------------------------------        
        return embeddings                                                     # (batch_size, input_seq_len, d_model)
        ## -------------------------------------------------------------------
##############################################################################################################


##############################################################################################################
class TransformerXLDecoderStack(tf.keras.Model):
    ## -------------------------------------------------------------------
    def __init__(self, num_layers, d_model, num_heads, dff, memory_length,
        max_sequence_length, enc_dec_attn=False, **kwargs):
        ## -------------------------------------------------------------------
        super(TransformerXLDecoderStack, self).__init__(**kwargs)
        ## -------------------------------------------------------------------
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.memory_length = tf.constant(memory_length)
        self.max_sequence_length = max_sequence_length
        self.enc_dec_attn = enc_dec_attn
        self.dropout_rate = 0.1
        ## -------------------------------------------------------------------
        self.dec_layers = [
            XLDecoderLayer(
                self.d_model,
                self.num_heads,
                self.dff,
                self.memory_length,
                self.max_sequence_length,
                rate = self.dropout_rate,
                enc_dec_attn = self.enc_dec_attn,
                layer_id = i + 1,
                name = 'XLDecoderLayer-{}'.format(i + 1)
            ) for i in range(self.num_layers)
        ]
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def call(self, embeddings, memory_length, positional_encoding, 
        padding_mask, encoder_outputs=None, training=None):
        ## -------------------------------------------------------------------
        outputx = embeddings
        for i in range(self.num_layers):
            outputx = self.dec_layers[i](
                outputx,
                memory_length,
                positional_encoding,
                padding_mask,
                encoder_outputs,
                training
            )
        ## -------------------------------------------------------------------
        return outputx                              # (batch_size, output_seq_len, d_model)
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------
##############################################################################################################
