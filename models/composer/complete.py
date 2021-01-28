import tensorflow as tf

class BERTransformer(tf.keras.Model):

  	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
				target_vocab_size, pe_input, pe_target, rate=0.1):
	    super(BERTransformer, self).__init__()

	    self.encoder = TransformerEncoderStack(
	    	num_layers, d_model, num_heads, dff, 
	    	input_vocab_size, pe_input, rate
		)

	    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  	def call(self, inputs, targets, enc_padding_mask, training=True):

    	enc_output = self.encoder(inp, training, enc_padding_mask)
    	# (batch_size, tar_seq_len, target_vocab_size)
    	final_output = self.final_layer(dec_output)
    	return final_output