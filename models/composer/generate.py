import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from .helpers.utils import create_RNN_cells, create_RNN_layer, create_segments


class RNNGenerator(tf.keras.Model):

    def __init__(self, configs):
        super(RNNGenerator, self).__init__()
        self.model = self.create_model(configs)
        self.model_path = configs['model_path']
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        initial_learning_rate = 0.001
        end_learning_rate = 0.00001
        decay_steps = 100000.0
        decay_rate = 0.
        learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
          initial_learning_rate, decay_steps, end_learning_rate, power=3
        )
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
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
        tune = tf.keras.Input(batch_input_shape = (16, configs['max_timesteps']))
        emb = tf.keras.layers.Embedding(
            input_dim = configs['vocab_size'],
            output_dim = configs['emb_size'],
            mask_zero = True
        )(tune)

        stacked_cells = tf.keras.layers.StackedRNNCells(
            create_RNN_cells(configs['rnn'])
        )

        self.sequential_RNN = create_RNN_layer(stacked_cells, stateful = True)
        rnn_output = self.sequential_RNN(emb)

        output = tf.keras.layers.Dense(configs['vocab_size'], activation='softmax')(rnn_output)
        model = tf.keras.Model(
            inputs=tune,
            outputs=output
        )
        model.summary()
        return model


    def loss_function(self, outputs,  targets, weighted = False):
        mask = tf.math.logical_not(tf.math.equal(outputs, 0))
        loss_ = self.loss_fn(
            y_pred = outputs, 
            y_true = targets
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    def run_step(self, sequence):
        inputs = sequence[:, :-1]
        targets = sequence[:, 1:]

        with tf.GradientTape() as tape:
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, outputs


    def train(self, dataset, configs):
        for epoch in range(configs['num_epochs']):
            for i, sequence in enumerate(dataset):

                losses = []
                curr_step = int(self.ckpt.step)
                self.model.reset_states()

                segments = create_segments(sequence)
                
                for segment in segments:
                    segment = tf.squeeze(segment)
                    loss_val, outputs = self.run_step(segment)
                    losses.append(loss_val)
                    print(loss_val)    
                    if curr_step % configs['print_every'] == 0: 
                        print(tf.argmax(outputs, axis=-1)[0])
                        print(segment[0])
                
                self.ckpt.step.assign_add(1)
                
                loss_val = tf.reduce_mean(losses).numpy()
                print('Loss: ' + str(loss_val))
                
                if curr_step % configs['save_every'] == 0:
                    self.update_tensorboard(loss_val, curr_step)
                
                if curr_step % 100 == 0:
                    self.save_model_checkpoint()


    def call(self, inputs):
        #print(inputs)
        return self.model(inputs)

    def predict(self, inputs):
        pass

class TransformerGenerator(tf.keras.Model):

    def __init__(self, configs):
        
        super(TransformerGenerator, self).__init__()
        self.model = self.create_model(configs)


    def create_model(self, configs):
        pass