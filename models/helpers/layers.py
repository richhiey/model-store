####################################################################
# ------------------- Helpers for building Transformers ------------
####################################################################
import numpy as np
import tensorflow as tf
####################################################################

## -------------------------------------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        # (batch_size, seq_len_q, num_heads, depth)        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(
            scaled_attention, 
            (batch_size, -1, self.d_model)
        )
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = tf.expand_dims(attn1, axis=1)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
        return out2, attn_weights_block1
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class Memory(keras.layers.Layer):
    """Positional embeddings.
    # Arguments
        batch_size: int > 0. Maximum batch size.
        memory_len: int > 0. Maximum memory length.
        target_len: int > 0. Maximum length of targets.
        output_dim: int > 0. Dimension of outputs.
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
        1D tensor with shape: `(batch_size,)` represents length of memory.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length + memory_length, output_dim)`.
    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self, batch_size, memory_len, target_len, output_dim, **kwargs):
        super(Memory, self).__init__(**kwargs)
        self.supports_masking = True
        self.stateful = True

        self.batch_size = batch_size
        self.memory_len = memory_len
        self.target_len = target_len
        self.output_dim = output_dim

        self.memory = None

    def build(self, input_shape):
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_len + self.target_len, self.output_dim),
            initializer='zeros',
            trainable=False,
            name='memory',
        )
        super(Memory, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], None, self.output_dim

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def call(self, inputs, **kwargs):
        inputs, memory_length = inputs
        memory_length = K.cast(memory_length[0][0], 'int32')
        batch_size = K.cast(K.shape(inputs)[0], 'int32')
        seq_len = K.cast(K.shape(inputs)[1], 'int32')

        # Build new memory
        pad = K.tile(inputs[0:1, ...], (self.batch_size - batch_size, 1, 1))
        padded = K.concatenate([inputs, pad], axis=0)              # (self.batch_size, seq_len, output_dim)
        new_memory = K.concatenate([self.memory, padded], axis=1)  # (self.batch_size, self.memory_len + seq_len, ...)
        new_memory = tf.slice(                                     # (self.batch_size, self.memory_len, output_dim)
            new_memory,
            (0, seq_len, 0),
            (self.batch_size, self.memory_len + self.target_len, self.output_dim),
        )
        self.add_update(K.update(self.memory, new_memory), inputs)

        # Build output
        old_memory = tf.slice(                                     # (batch_size, memory_length, output_dim)
            new_memory,
            (0, K.maximum(0, self.memory_len + self.target_len - seq_len - memory_length), 0),
            (batch_size, K.minimum(self.memory_len, memory_length), self.output_dim),
        )

        return old_memory

    def get_config(self):
        config = {
            'batch_size': self.batch_size,
            'memory_len': self.memory_len,
            'target_len': self.target_len,
            'output_dim': self.output_dim,
        }
        base_config = super(Memory, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
## -------------------------------------------------------------------


## -------------------------------------------------------------------
class RelativePartialMultiHeadSelfAttention(keras.layers.Layer):
    """Positional embeddings.
    # Arguments
        units: int >= 0. Dimensions of all tensors.
        num_head: int >= 0. Number of heads. Should divide units.
        use_bias: Boolean. Whether to use bias term.
        attention_dropout: 0.0 < float < 1.0. Dropout rate for attention weights.
    # Input shape
        First 3D tensor with shape: `(batch_size, sequence_length, units)`.
        Second 3D tensor with shape: `(batch_size, previous_sequence_length + sequence_length, units)`.
        Third 3D tensor with shape: `(batch_size, previous_sequence_length, units)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, units)`.
    # References
        - [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf)
    """

    def __init__(self,
                 units,
                 num_head,
                 activation=None,
                 use_bias=False,
                 attention_dropout=0.0,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(RelativePartialMultiHeadSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.num_head = num_head
        self.units_head = units // num_head
        self.activation = activation
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel_q, self.kernel_kv, self.kernel_o, self.kernel_r = (None,) * 4
        self.bias_q, self.bias_kv, self.bias_o, self.bias_r = (None,) * 4
        self.att_drop_layer = None

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask[0]
        return None

    def build(self, input_shape):
        self.kernel_q = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_q',
        )
        if self.use_bias:
            self.bias_q = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_q',
            )

        self.kernel_kv = self.add_weight(
            shape=(self.units, self.units * 2),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_kv',
        )
        if self.use_bias:
            self.bias_kv = self.add_weight(
                shape=(self.units * 2,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_kv',
            )

        self.kernel_o = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_o',
        )
        if self.use_bias:
            self.bias_o = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_o',
            )

        self.kernel_r = self.add_weight(
            shape=(self.units, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel_r',
        )
        if self.use_bias:
            self.bias_r = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias_r',
            )
        if 0.0 < self.attention_dropout < 1.0:
            self.att_drop_layer = keras.layers.Dropout(self.attention_dropout)
        super(RelativePartialMultiHeadSelfAttention, self).build(input_shape)

    def _reshape_to_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size, seq_len, self.num_head, self.units_head))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * self.num_head, seq_len, self.units_head))

    def _reshape_from_batches(self, x):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // self.num_head, self.num_head, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // self.num_head, seq_len, feature_dim * self.num_head))

    def _reshape_mask(self, mask):
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, self.num_head, 1])
        return K.reshape(mask, (-1, seq_len))

    @staticmethod
    def _relative_shift(x):
        batch_size, q_len, k_len = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2]
        x = tf.pad(x, [[0, 0], [0, 0], [1, 0]])               # (batch * n_head, seq_len, prev_len + seq_len + 1)
        x = K.reshape(x, (batch_size, k_len + 1, q_len))      # (batch * n_head, prev_len + seq_len + 1, seq_len)
        x = x[:, 1:, :]                                       # (batch * n_head, prev_len + seq_len, seq_len)
        return K.reshape(x, (batch_size, q_len, k_len))       # (batch * n_head, seq_len, prev_len + seq_len)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, mask=None, training=None):
        inputs, relatives, memories, bias_context, bias_relative = inputs
        full = K.concatenate([memories, inputs], axis=1)      # (batch, prev_len + seq_len, units)
        w_q = K.dot(inputs, self.kernel_q)                    # (batch, seq_len, units)
        w_kv = K.dot(full, self.kernel_kv)                    # (batch, prev_len + seq_len, units * 2)
        w_r = K.dot(relatives, self.kernel_r)                 # (batch, prev_len + seq_len, units)
        if self.use_bias:
            w_q = K.bias_add(w_q, self.bias_q)
            w_kv = K.bias_add(w_kv, self.bias_kv)
            w_r = K.bias_add(w_r, self.bias_r)
        if self.activation is not None:
            w_q = self.activation(w_q)
            w_kv = self.activation(w_kv)
            w_r = self.activation(w_r)

        w_k = w_kv[:, :, :self.units]                         # (batch, prev_len + seq_len, units)
        w_v = w_kv[:, :, self.units:]                         # (batch, prev_len + seq_len, units)

        w_qc = K.bias_add(w_q, bias_context)
        w_qc = self._reshape_to_batches(w_qc)                 # (batch * n_head, seq_len, units_head)
        w_k = self._reshape_to_batches(w_k)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_context = K.batch_dot(w_qc, w_k, axes=2)            # (batch * n_head, seq_len, prev_len + seq_len)

        w_qr = K.bias_add(w_q, bias_relative)
        w_qr = self._reshape_to_batches(w_qr)                 # (batch * n_head, seq_len, units_head)
        w_r = self._reshape_to_batches(w_r)                   # (batch * n_head, prev_len + seq_len, units_head)
        a_relative = K.batch_dot(w_qr, w_r, axes=2)           # (batch * n_head, seq_len, prev_len + seq_len)
        a_relative = self._relative_shift(a_relative)         # (batch * n_head, seq_len, prev_len + seq_len)

        att = (a_context + a_relative) / K.sqrt(K.constant(self.units_head, dtype=K.floatx()))
        exp = K.exp(att - K.max(att, axis=-1, keepdims=True))

        q_len, k_len = K.shape(w_q)[1], K.shape(w_k)[1]
        indices = K.expand_dims(K.arange(0, k_len), axis=0)
        upper = K.expand_dims(K.arange(k_len - q_len, k_len), axis=-1)
        exp *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None and mask[0] is not None:
            mask = K.cast(mask[0], K.floatx())
            mask = K.concatenate([K.ones_like(memories[:, :, 0]), mask], axis=1)
            exp *= K.expand_dims(self._reshape_mask(mask), axis=1)

        att = exp / K.sum(exp, axis=-1, keepdims=True)
        if self.att_drop_layer is not None:
            att = self.att_drop_layer(att, training=training)
        w_v = self._reshape_to_batches(w_v)                   # (batch * n_head, prev_len + seq_len, units_head)
        w_o = K.batch_dot(att, w_v)                           # (batch * n_head, seq_len, units_head)

        w_o = self._reshape_from_batches(w_o)                 # (batch, seq_len, units)
        w_o = K.dot(w_o, self.kernel_o)                       # (batch, seq_len, units)
        if self.use_bias:
            w_o = K.bias_add(w_o, self.bias_o)
        if self.activation is not None:
            w_o = self.activation(w_o)

        if TF_KERAS:
            # Add shape information to tensor when using `tf.keras`
            input_shape = K.int_shape(inputs)
            if input_shape[1] is not None:
                w_o = K.reshape(w_o, (-1,) + input_shape[1:])
        return w_o

    def get_config(self):
        config = {
            'units': self.units,
            'num_head': self.num_head,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'attention_dropout': self.attention_dropout,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(RelativePartialMultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
## -------------------------------------------------------------------

####################################################################
