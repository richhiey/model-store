import tensorflow as tf
import numpy as np
import os
from datetime import datetime

class Generator(tf.keras.Model):

    def __init__(self, configs):
        super(Generator, self).__init__()
        self.model = self.create_model(configs)
        self.model_path = configs['model_path']
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.tensorboard_logdir = os.path.join(
            self.model_path,
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir)
        )
        self.file_writer.set_as_default()
        self.ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.optimizer,
            net = self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(self.model_path, 'ckpt'),
            max_to_keep = 3
        )
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def save_model_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))


    def update_tensorboard(self, loss, step, grads=None):
        with self.file_writer.as_default():
            tf.summary.scalar("Categorical Cross-Entropy", loss, step=step)
        self.file_writer.flush()


    def create_model(self, configs):
        tune = tf.keras.Input(batch_input_shape = (8, configs['max_timesteps']))
        emb = tf.keras.layers.Embedding(
            input_dim = configs['vocab_size'],
            output_dim = configs['emb_size']
        )(tune)

        lstm_output_1 = tf.keras.layers.LSTM(configs['lstm_units'], return_sequences=True, stateful=True)(emb)
        lstm_output_2 = tf.keras.layers.LSTM(configs['lstm_units'], return_sequences=True, stateful=True)(lstm_output_1)
        #lstm_output_3 = tf.keras.layers.LSTM(configs['lstm_units'], return_sequences=True, stateful=True)(lstm_output_2)

        dense = tf.keras.layers.Dense(configs['dense_units'], activation='sigmoid')(lstm_output_2)
        output = tf.keras.layers.Dense(configs['vocab_size'], activation='softmax')(dense)
        model = tf.keras.Model(
            inputs=tune,
            outputs=output
        )
        model.summary()
        return model


    def run_step(self, sequence):
        inputs = sequence[:, :-1]
        targets = sequence[:, 1:]
        with tf.GradientTape() as tape:
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, outputs


    def create_segments(self, sequence, seg_len=128):
        seq_shape = tf.shape(sequence)
        print(seq_shape)
        num_splits = int(seq_shape[2] / seg_len)
        segments = tf.split(sequence[:, :, :seg_len*num_splits], num_or_size_splits=num_splits, axis=-1)
        #print(segments)
        return segments


    def train(self, dataset, configs):
        for epoch in range(configs['num_epochs']):
            for sequence in dataset:
                segments = self.create_segments(sequence)
                self.model.reset_states()
                losses = []
                for segment in segments:
                    segment = tf.squeeze(segment)
                    loss_val, outputs = self.run_step(segment)
                    losses.append(loss_val)
                loss_val = tf.reduce_mean(losses).numpy()
                print('Loss: ' + str(loss_val))
                self.ckpt.step.assign_add(1)
                curr_step = tf.cast(self.ckpt.step, tf.int64)
                self.update_tensorboard(loss_val, curr_step)
                
                if curr_step % 100:
                    self.save_model_checkpoint()


    def call(self, inputs):
        #print(inputs)
        return self.model(inputs)