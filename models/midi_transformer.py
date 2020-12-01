## MIDI Transformer on Pop909 dataset

import os
import json
import tensorflow as tf

from datetime import datetime
from .helpers.blocks import TransformerXLEncoderStack, \
                            TransformerXLDecoderStack
from .helpers.utils import  positional_encoding, \
                            create_look_ahead_mask


class MIDITransformer(tf.keras.Model):
## -------------------------------------------------------------------
    def __init__(self, config_path, model_path, **kwargs):

        super(MIDITransformer, self).__init__(**kwargs)
        ## -------------------------------------------------------------------
        self.model_path = model_path
        if os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        ## -------------------------------------------------------------------
        self.config_path = config_path
        with open(self.config_path) as json_file: 
            self.configs = json.loads(json_file.read())
        ## -------------------------------------------------------------------
        self.saved_model_dir = os.path.join(self.model_path, 'MIDI_Transformer')
        self.initial_learning_rate = 0.001
        self.end_learning_rate = 0.00001
        self.decay_steps = 100000.0
        self.decay_rate = 0.
        self.max_sequence_length = int(self.configs['encoder']['max_sequence_length'])
        self.d_model = int(self.configs['encoder']['d_model'])
        self.batch_size = int(self.configs['batch_size'])
        self.input_vocab_size = int(self.configs['encoder']['input_vocab_size'])

        learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
          self.initial_learning_rate, self.decay_steps, self.end_learning_rate, power=3
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ## -------------------------------------------------------------------
        self.pos_encoding = positional_encoding(
            2 * self.max_sequence_length,
            self.d_model
        )
        self.input_embedding_layer = tf.keras.layers.Embedding(
            self.input_vocab_size,
            self.d_model,
            name='input_midi_embedding'
        )
        self.output_embedding_layer = tf.keras.layers.Embedding(
            self.target_vocab_size,
            self.d_model,
            name='output_midi_embedding'
        )
        self.encoder_stack = TransformerXLEncoderStack(self.configs['encoder'])
        self.decoder_stack = TransformerXLDecoderStack(self.configs['decoder'])
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        self.decoder_optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        ## -------------------------------------------------------------------
        self.encoder_ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.encoder_optimizer,
            net = self.encoder_stack
        )
        self.encoder_ckpt_manager = tf.train.CheckpointManager(
            self.encoder_ckpt, 
            os.path.join(self.model_path, 'encoder', 'ckpt'),
            max_to_keep = 3
        )
        self.encoder_ckpt.restore(self.encoder_ckpt_manager.latest_checkpoint)
        if self.encoder_ckpt_manager.latest_checkpoint:
            print("Restored Encoder from {}".format(self.encoder_ckpt_manager.latest_checkpoint))
        else:
            print("Initializing Encoder from scratch.")
        ## -------------------------------------------------------------------
        self.decoder_ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.decoder_optimizer,
            net = self.decoder_stack
        )
        self.decoder_ckpt_manager = tf.train.CheckpointManager(
            self.decoder_ckpt, 
            os.path.join(self.model_path, 'decoder', 'ckpt'),
            max_to_keep = 3
        )
        self.decoder_ckpt.restore(self.decoder_ckpt_manager.latest_checkpoint)
        if self.decoder_ckpt_manager.latest_checkpoint:
            print("Restored Decoder from {}".format(self.decoder_ckpt_manager.latest_checkpoint))
        else:
            print("Initializing Decoder from scratch.")            
        ## -------------------------------------------------------------------
        self.tensorboard_logdir = os.path.join(
            self.model_path,
            'tensorboard',
            'run'+datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_logdir, 'metrics')
        )
        self.file_writer.set_as_default()
        ## -------------------------------------------------------------------


    def run_step(self, inputs, targets, pe, look_ahead_mask, memory_length):
        input_embeddings = self.input_embedding_layer(inputs)
        encoder_output = self.encoder_stack([input_embeddings, memory_length], pe)
        print('Encoder output - ')
        print(tf.shape(encoder_output))
        print('Decoder output - ')
        output_embeddings = self.output_embedding_layer(targets)
        decoder_output, attn_weights = self.decoder_stack([output_embeddings, memory_length], encoder_output, pe, look_ahead_mask)
        print(tf.shape(decoder_output))
        return decoder_output, attn_weights

    
    def call(self, inputs, targets, positional_encoding, look_ahead_mask, memory_length):
        return self.run_step(inputs, targets, positional_encoding, look_ahead_mask, memory_length)


    def reset_model_state(self):
        self.encoder_stack.reset_states()
        self.decoder_stack.reset_states()


    def save_model_checkpoint(self):
        encoder_save_path = self.encoder_ckpt_manager.save()
        decoder_save_path = self.decoder_ckpt_manager.save()
        print(
            "Saved checkpoint for step {}: {}, {}".format(
                int(self.ckpt.step),
                encoder_save_path,
                decoder_save_path
            )
        )


    def calculate_loss(self, outputs,  targets, weighted = False):
        mask = tf.math.logical_not(tf.math.equal(outputs, 0))
        loss_ = self.cross_entropy(
            y_pred = outputs, 
            y_true = targets
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    def train_step(self, inputs, targets, look_ahead_mask):
        with tf.GradientTape() as tape:
            outputs, attn_weights = run_step(
                inputs, 
                targets, 
                self.pos_encoding, 
                look_ahead_mask, 
                tf.constant(self.max_sequence_length)
            )
            targets = tf.reshape(tf.sparse.to_dense(targets), (-1, 255))
            loss_value = self.calculate_loss(
                outputs = outputs,
                targets = targets
            )
            print(loss_value)
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        gradients = [(tf.clip_by_norm(grad, 3.0)) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss_value, outputs, attn_weights


    def train(self, dataset, train_configs):
        
        look_ahead_mask = create_look_ahead_mask(self.max_sequence_length)

        for epoch in range(configs['num_epochs']):
            # Training loop
            for i, song in enumerate(dataset):
                # Optimize the model
                self.reset_model_state()

                inputs = tf.split(song['MELODY'], num_or_size_splits=self.max_sequence_length, axis=1)
                outputs = tf.split(song['PIANO'], num_or_size_splits=self.max_sequence_length, axis=1)

                for j, x in enumerate(inputs):
                    loss_value, outputs, attn_weights = self.train_step(x, outputs[j], look_ahead_mask)
                    print(loss_value)

                self.encoder_ckpt.step.assign_add(1)
                self.decoder_ckpt.step.assign_add(1)

                if i % 100 is 0:
                    self.save_model_checkpoint()
            
            self.save_model_checkpoint()