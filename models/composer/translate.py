import tensorflow as tf

class Translator(tf.keras.Model):

	def __init__(self):
		self.model = self.create_model()

	def create_model(self):
		pass
