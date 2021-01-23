import tensorflow as tf

class Completor(tf.keras.Model):

	def __init__(self):
		self.model = self.create_model()

	def create_model(self):
		pass