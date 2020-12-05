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
    ## -------------------------------------------------------------------
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
        self.num_layers = int(self.configs['num_layers'])
        self.max_sequence_length = int(self.configs['max_sequence_length'])
        self.d_model = int(self.configs['d_model'])
        self.batch_size = int(self.configs['batch_size'])
        self.vocab_size = int(self.configs['vocab_size'])
        self.memory_length = int(self.configs['memory_length'])
        self.task_name = self.configs['task_name']
        self.initial_learning_rate = int(self.configs['initial_learning_rate']) # 0.001
        self.end_learning_rate = int(self.configs['end_learning_rate']) # 0.00001
        self.decay_steps = int(self.configs['decay_steps']) # 1000
        ## -------------------------------------------------------------------
        self.learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
          self.initial_learning_rate, self.decay_steps, self.end_learning_rate, power=3
        )
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate_fn)
        ## -------------------------------------------------------------------
        self.posîtional_encoding = tf.tile(
            positional_encoding(
                self.memory_length * self.max_sequence_length, self.d_model
            ),
            [self.batch_size, 1, 1]
        )
        ## -------------------------------------------------------------------
        if bool(self.configs.get('create_encoder')):
            self.encoder_embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.d_model,
                name='encoder_embedding'
            )
            self.encoder_stack = TransformerXLEncoderStack(self.configs['encoder'])
            self.enc_ckpt, self.enc_ckpt_manager = self.create_or_restore_checkpoint(
                'encoder', self.encoder_stack, self.optimizer
            )
        ## -------------------------------------------------------------------
        if bool(self.configs.get('create_decoder')):
            self.decoder_embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.d_model,
                name='decoder_embedding'
            )
            self.decoder_stack = TransformerXLDecoderStack(self.configs['decoder'])
            self.dec_ckpt, self.dec_ckpt_manager = self.create_or_restore_checkpoint(
                'decoder', self.decoder_stack, self.optimizer
            )
        ## -------------------------------------------------------------------
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)
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
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def train(self, dataset, train_configs=None):
        ## -------------------------------------------------------------------
        preprocessing_fn = eval('self.' + self.task_name + '_preprocess')
        train_fn = eval('self.' + self.task_name)
        loss_fn = eval('self.' + self.task_name + '_loss')
        # assert train_configs.get('task_name')
        inputs_key = train_configs.get('inputs')
        # assert train_configs.get('task_name')
        outputs_key = train_configs.get('outputs')
        print('Training the encoder on key: ' + inputs_key)
        print('Training the encoder on key: ' + outputs_key)
        # -------------------------------------------------------------------
        for epoch in range(int(train_configs['num_epochs'])):
            # -------------------------------------------------------------------
            for i, song in enumerate(dataset):
                ## -------------------------------------------------------------------
                ## RUN TRAINING TASK
                ## -------------------------------------------------------------------                
                input_segments, \
                output_segments, \
                train_mask, padding_mask = preprocessing_fn(
                    song[inputs_key], inputs_key
                )
                loss_value, outputs = self.train_step(
                    train_fn, loss_fn, input_segments, output_segments,
                    self.positional_encoding, train_mask, padding_mask
                )
                ## -------------------------------------------------------------------
                print('Loss (' + str(i) + ') - ' + str(loss_value))
                tf.summary.scalar('Cross Entropy Loss', loss_value, step=int(self.encoder_ckpt.step))
                if i % 100 == 0:
                    print(outputs)
                    self.save_model_checkpoint()
                       
                self.encoder_ckpt.step.assign_add(1)
                self.decoder_ckpt.step.assign_add(1)
                ## -------------------------------------------------------------------
            self.save_model_checkpoint()
            # -------------------------------------------------------------------
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def train_step(
        self, model_fn, loss_fn, inputs, outputs,
        positional_encoding, train_mask, padding_mask
    ):
        ## -------------------------------------------------------------------
        self.reset_model_state()
        all_vars = [self.encoder_stack.trainable_variables, self.decoder_stack.trainable_variables]
        flat_list_vars = [item for sublist in all_vars for item in sublist]
        memory_length = tf.constant(train_configs['memory_length'])
        ## -------------------------------------------------------------------
        with tf.GradientTape() as tape:
        ## -------------------------------------------------------------------
            outputs, model_vars = model_fn(inputs, outputs, positional_encoding, train_mask, padding_mask)
            final_output = self.output_dropout(
                self.output_layer(decoder_output),
                training=True
            )
            final_softmax = tf.nn.softmax(decoder_output)
            loss_value = loss_fn(outputs, final_softmax)       
            gradients = tape.gradient(loss_value, model_vars)
        ## -------------------------------------------------------------------
        self.optimizer.apply_gradients(zip(gradients, model_vars))
        ## -------------------------------------------------------------------
        return loss_value, outputs
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    ## TASK - MUSIC GENERATION (DECODER ONLY)
    ## -------------------------------------------------------------------
    def music_generation_preprocess(self, sequence, key):
        pass
    ## -------------------------------------------------------------------
    def music_generation(self, inputs, ouputs, pe, t_mask, p_mask):
        return run_decoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask,
            encoder_outputs=None, training=None
        )
    ## -------------------------------------------------------------------
    def music_generation_loss(self, sequence, key):
        pass
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    ## TASK - MUSIC INPAINTING (ENCODER ONLY)
    ## -------------------------------------------------------------------
    def music_inpainting(self):
        pass
    ## -------------------------------------------------------------------
    def music_inpainting_preprocess(self, inputs, ouputs, pe, t_mask, p_mask):
        return run_encoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask, training=True
        )
    ## -------------------------------------------------------------------
    def music_inpainting_loss(self, sequence, key):
        pass
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    ## TASK - MUSIC TRANSLATION (ENCODER + DECODER)
    ## -------------------------------------------------------------------
    def musical_translation_preprocess(self, song, input_key, output_key):
        pass
    ## -------------------------------------------------------------------
    def musical_translation(self, inputs, ouputs, pe, t_mask, p_mask):
        encoder_outputs = run_encoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask, training=True
        )

        return run_decoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask,
            encoder_outputs=encoder_outputs, training=True
        )
    ## -------------------------------------------------------------------
    def music_translation_loss(self, sequence, key):
        pass
    ## -------------------------------------------------------------------


    # ------------------------------------------------------------------------------        
    def create_musical_segments(self, inputs, key):
        # ------------------------------------------------------------------------------
        inp_pad_size = 0
        inputs = tf.sparse.to_dense(inputs)
        segment_length = tf.shape(inputs)[1]
        if (segment_length % self.max_sequence_length):
            inp_pad_size = int(segment_length / self.max_sequence_length)
            inp_pad_size = ((inp_pad_size + 1) * self.max_sequence_length) - segment_length
            inputs = tf.pad(inputs, [[0, 0],[0, inp_pad_size]])
        # ------------------------------------------------------------------------------        
        num_splits = math.ceil(segment_length / self.max_sequence_length)
        return tf.split(
            inputs,
            num_or_size_splits=num_splits, 
            axis=1
        )
        # ------------------------------------------------------------------------------        
    # ------------------------------------------------------------------------------        


    ## -------------------------------------------------------------------
    def call(self, inputs, targets, positional_encoding, memory_length):
        return self.run_step(inputs, targets, positional_encoding, memory_length)
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def reset_model_state(self):
        self.encoder_stack.reset_states()
        self.decoder_stack.reset_states()
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def run_encoder_stack(
        self, inputs, positional_encoding, memory_length, t_mask, padding_mask,
        training=True
    ):
        ## -------------------------------------------------------------------
        embeddings = self.encoder_embedding(inputs)
        return self.encoder_stack(
            embeddings,                 # Query
            embeddings,                 # Key
            embeddings,                 # Value
            memory_length               # Length of previous memories
            positional_encoding,        # Sinusoidal positional encoding
            t_mask,                     # Task related mask (Look ahead mask or Masked-LM mask)
            padding_mask                # Padding mask for input
            training,                   # Mode of operation
        )
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def run_decoder_stack(
        self, inputs, positional_encoding, memory_length, t_mask, padding_mask, 
        encoder_outputs=None, training=None
    ):
        ## -------------------------------------------------------------------
        embeddings = self.decoder_embedding(inputs)
        query = inputs
        if (encoder_outputs):
            key = encoder_outputs
        else:
            key = inputs
        value = key
        return self.decoder_stack(
            query,                      # Query
            key,                        # Key
            value,                      # Value
            memory_length               # Length of memory of previous segments
            positional_encoding,        # Sinusoidal positional encoding
            t_mask,                     # Task related mask (Look ahead mask or Masked-LM mask)
            padding_mask                # Padding mask for input
            training,                   # Mode of operation
        )
        ## -------------------------------------------------------------------
        return outputs, attn_weights
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def create_or_restore_checkpoint(self, name, model, optimizer):
        ## -------------------------------------------------------------------
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=self.encoder_stack)
        ckpt_path = os.path.join(self.model_path, 'ckpt', name)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_path max_to_keep=3)
        self.ckpt.restore(self.encoder_ckpt_manager.latest_checkpoint)
        ## -------------------------------------------------------------------
        if self.encoder_ckpt_manager.latest_checkpoint:
            print("Restored" + name.upper() + "from {}".format(self.encoder_ckpt_manager.latest_checkpoint))
        else:
            print("Initializing" + name.upper() + "from scratch.")
        return self.ckpt, self.ckpt_manager
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def save_model_checkpoint(self):
        ## -------------------------------------------------------------------
        encoder_save_path = self.encoder_ckpt_manager.save()
        decoder_save_path = self.decoder_ckpt_manager.save()
        print(
            "Saved checkpoint for step {}: {}, {}".format(
                int(self.encoder_ckpt.step),
                encoder_save_path,
                decoder_save_path
            )
        )
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------
