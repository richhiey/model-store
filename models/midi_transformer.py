## MIDI Transformer on Pop909 dataset

import os
import json
import tensorflow as tf
from datetime import datetime
from .helpers.blocks import TransformerXLEncoderStack, \
                            TransformerXLDecoderStack
from .helpers.utils import  positional_encoding, \
                            create_look_ahead_mask, \
                            create_padding_mask


class MIDITransformer(tf.keras.Model):

    def __init__(self, config_path, model_path, **kwargs):
        ## -------------------------------------------------------------------
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
        self.target_vocab_size = int(self.configs['decoder']['target_vocab_size'])

        learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
          self.initial_learning_rate, self.decay_steps, self.end_learning_rate, power=3
        )
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
        self.output_layer = tf.keras.layers.Dense(389)
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


    def run_step(self, padded_inputs, padded_targets, positional_encoding, memory_length, training=True):
        ## -------------------------------------------------------------------
        positional_encoding = tf.tile(positional_encoding, [tf.shape(padded_inputs)[0], 1, 1])
        self.reset_model_state()

        for x in tf.split(padded_inputs, num_or_size_splits=int(tf.shape(padded_inputs)[1]/self.max_sequence_length), axis=1):
            input_embeddings = self.input_embedding_layer(x)
            ## -------------------------------------------------------------------
            encoder_output = self.encoder_stack(
                [input_embeddings, memory_length],
                positional_encoding,
                training,
                create_padding_mask(x)
            )
        ## -------------------------------------------------------------------
        decoder_outputs = []
        for y in tf.split(padded_targets, num_or_size_splits=int(tf.shape(padded_targets)[1]/self.max_sequence_length), axis=1):
            output_embeddings = self.output_embedding_layer(y)
            decoder_output, attn_weights = self.decoder_stack(
                [output_embeddings, memory_length],
                encoder_output,
                positional_encoding,
                training,
                create_padding_mask(y)
            )
            decoder_output = self.output_layer(decoder_output)
            final_softmax = tf.nn.softmax(decoder_output)
            decoder_outputs.append(final_softmax)
        decoder_outputs = tf.concat(decoder_outputs, axis=1)
        ## -------------------------------------------------------------------
        return decoder_outputs, attn_weights
        ## -------------------------------------------------------------------

    
    def call(self, inputs, targets, positional_encoding, memory_length):
        ## -------------------------------------------------------------------
        return self.run_step(inputs, targets, positional_encoding, memory_length)
        ## -------------------------------------------------------------------


    def reset_model_state(self):
        ## -------------------------------------------------------------------
        self.encoder_stack.reset_states()
        self.decoder_stack.reset_states()
        ## -------------------------------------------------------------------


    def save_model_checkpoint(self):
        ## -------------------------------------------------------------------
        encoder_save_path = self.encoder_ckpt_manager.save()
        decoder_save_path = self.decoder_ckpt_manager.save()
        ## -------------------------------------------------------------------
        print(
            "Saved checkpoint for step {}: {}, {}".format(
                int(self.encoder_ckpt.step),
                encoder_save_path,
                decoder_save_path
            )
        )
        ## -------------------------------------------------------------------


    def calculate_loss(self, outputs,  targets, weighted = False):
        mask = tf.math.logical_not(tf.math.equal(outputs, 0))
        loss_ = self.cross_entropy(
            y_pred = outputs, 
            y_true = targets
        )
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    def train_step(self, inputs, targets):
        ## -------------------------------------------------------------------
        if (tf.shape(inputs)[1] % self.max_sequence_length):
            inp_pad_size = int(tf.shape(inputs)[1] / self.max_sequence_length)
            inp_pad_size = (inp_pad_size + 1) * self.max_sequence_length - tf.shape(inputs)[1]
            inputs = tf.pad(inputs, [[0, 0],[0, inp_pad_size]])

        if (tf.shape(targets)[1] % self.max_sequence_length):
            tar_pad_size = int(tf.shape(targets)[1] / self.max_sequence_length)
            tar_pad_size = (tar_pad_size + 1) * self.max_sequence_length - tf.shape(targets)[1]
            targets = tf.pad(targets, [[0, 0],[0, tar_pad_size]])

        with tf.GradientTape() as tape:
            outputs, attn_weights = self.run_step(
                inputs, 
                targets, 
                self.pos_encoding, 
                tf.constant(self.max_sequence_length),
            )
            loss_value = self.calculate_loss(outputs = outputs, targets = targets)
        all_vars = [self.encoder_stack.trainable_variables, self.decoder_stack.trainable_variables]
        flat_list_vars = [item for sublist in all_vars for item in sublist]
        ## -------------------------------------------------------------------
        gradients = tape.gradient(loss_value, flat_list_vars)
        ## -------------------------------------------------------------------
        self.encoder_optimizer.apply_gradients(
            zip(gradients, flat_list_vars)
        )
        ## -------------------------------------------------------------------
        return loss_value, outputs, attn_weights
        ## -------------------------------------------------------------------


    def train(self, dataset, train_configs=None):
        ## -------------------------------------------------------------------
        for epoch in range(1000):
            ## -------------------------------------------------------------------
            for i, song in enumerate(dataset):
                ## -------------------------------------------------------------------
                self.reset_model_state()
                ## -------------------------------------------------------------------
                melody = tf.sparse.to_dense(song['melody'])
                rhythm = tf.sparse.to_dense(song['rhythm'])
                ## -------------------------------------------------------------------
                loss_value, outputs, attn_weights = self.train_step(melody, rhythm)
                print('Loss (' + str(i) + ') - ' + str(loss_value))
                ## -------------------------------------------------------------------
                tf.summary.scalar('Cross Entropy Loss', loss_value, step=int(self.encoder_ckpt.step))
                self.encoder_ckpt.step.assign_add(1)
                self.decoder_ckpt.step.assign_add(1)

                if i % 100 == 0:
                    self.save_model_checkpoint()
                ## -------------------------------------------------------------------
            self.save_model_checkpoint()
            ## -------------------------------------------------------------------
