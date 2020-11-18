## MIDI Transformer on Pop909 dataset

import tensorflow as tf
from helpers.layers import TransformerEncoder, TransformerDecoder
class MIDITransformer(tf.keras.Model):

	def __init__(self):
		self.midi_encoder = TransformerEncoder()
		self.midi_decoder = TransformerDecoder()

	def __create_model__(self):
		

