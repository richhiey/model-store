## MIDI Transformer on Pop909 dataset

import os
import json
import math
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
from .helpers.blocks import TransformerXLEncoderStack, \
                            TransformerXLDecoderStack
from .helpers.utils import  positional_encoding, \
                            create_look_ahead_mask, \
                            create_padding_mask, \
                            reconstruct_and_play_audio

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        tf.summary.scalar('Learning Rate', lr, step=int(step))
        return lr


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
        self.max_sequence_length = int(self.configs['max_sequence_length']) - 1
        self.memory_length = int(self.configs['memory_length'])
        self.num_layers = int(self.configs['num_layers'])
        self.d_model = int(self.configs['d_model'])
        self.num_heads = int(self.configs['num_heads'])
        self.dff = int(self.configs['dff'])
        self.batch_size = int(self.configs['batch_size'])
        self.vocab_size = int(self.configs['vocab_size'])
        self.task_name = self.configs['task_name']
        self.initial_learning_rate = float(self.configs['initial_learning_rate']) # 0.001
        self.end_learning_rate = float(self.configs['end_learning_rate']) # 0.00001
        self.decay_steps = int(self.configs['decay_steps']) # 1000
        self.create_encoder = bool(self.configs.get('create_encoder'))
        self.create_decoder = bool(self.configs.get('create_decoder'))
        ## -------------------------------------------------------------------
        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        ## -------------------------------------------------------------------
        self.positional_encoding = tf.tile(
            positional_encoding(
                self.memory_length + self.max_sequence_length, self.d_model
            ),
            [self.batch_size, 1, 1]
        )
        ## -------------------------------------------------------------------
        if self.create_encoder:
            self.encoder_embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.d_model,
                name='encoder_embedding'
            )
            self.enc_dropout = tf.keras.layers.Dropout(0.1)
            self.encoder_stack = TransformerXLEncoderStack(
                self.num_layers, self.d_model, self.num_heads, self.dff,
                self.memory_length, self.max_sequence_length
            )
            self.enc_ckpt, self.enc_ckpt_manager = self.create_or_restore_checkpoint(
                'encoder', self.encoder_stack, self.optimizer
            )
        ## -------------------------------------------------------------------
        if self.create_decoder:
            self.decoder_embedding = tf.keras.layers.Embedding(
                self.vocab_size, self.d_model,
                name='decoder_embedding'
            )
            self.dec_dropout = tf.keras.layers.Dropout(0.1)
            self.decoder_stack = TransformerXLDecoderStack(
                self.num_layers, self.d_model, self.num_heads, self.dff,
                self.memory_length, self.max_sequence_length,
                enc_dec_attn = self.create_encoder 
            )
            self.dec_ckpt, self.dec_ckpt_manager = self.create_or_restore_checkpoint(
                'decoder', self.decoder_stack, self.optimizer
            )
        ## -------------------------------------------------------------------
        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        self.output_dropout = tf.keras.layers.Dropout(0.1)
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
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
            for i, tracks in tqdm(enumerate(dataset)):
                ## -------------------------------------------------------------------
                ## RUN TRAINING TASK
                ## -------------------------------------------------------------------                
                current_step = i
                input_segments, \
                output_segments, \
                padding_mask = preprocessing_fn(tracks, inputs_key, outputs_key)
                #print('------------- INPUT SEGMENTS -------------')
                #print(input_segments)
                #print('------------- OUTPUT SEGMENTS -------------')
                #print(output_segments)
                #print('------------- PADDING MASK -------------')
                #print(padding_mask)
                loss_value, outputs = self.train_step(
                    train_fn, loss_fn, input_segments, output_segments,
                    self.positional_encoding, padding_mask, current_step
                )
                ## -------------------------------------------------------------------
                if i % 1000 == 0:
                    print('Lets listen to what to the model sounds like step ' + str(current_step) + '!')
                    reconstruct_and_play_audio(input_segments)
                    reconstruct_and_play_audio(output_segments)
                    reconstruct_and_play_audio(outputs)
                if i % 100 == 0:
                    print('Loss (' + str(i) + ') - ' + str(loss_value))
                    print('Predicted by model:')
                    print(tf.math.argmax(outputs, axis=-1))
                    print('Real Music:')
                    print(tf.concat(input_segments, axis=-1))
                    self.save_model_checkpoint()
                if self.create_encoder:                       
                    self.enc_ckpt.step.assign_add(1)
                if self.create_decoder:
                    self.dec_ckpt.step.assign_add(1)
                ## -------------------------------------------------------------------
            self.save_model_checkpoint()
            # -------------------------------------------------------------------
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    #@tf.function
    def train_step(
        self, model_fn, loss_fn, inputs, targets,
        positional_encoding, padding_mask, current_step
    ):
        ## -------------------------------------------------------------------
        self.reset_model_state()
        memory_length = tf.constant(self.memory_length)
        ## -----------l--------------------------------------------------------
        num_segments = 0
        loss_values = []
        all_outputs = []
        ## -----------l--------------------------------------------------------
        for x, y, z in zip(inputs, targets, padding_mask):
        ## -------------------------------------------------------------------
            with tf.GradientTape() as tape:
                if tf.equal(tf.reduce_sum(x), 0):
                    print('ALL INPUTS ARE PADDED, SO WE WILL SKIP THIS ONE!!')
                else:
                    outputs = model_fn(x, y, positional_encoding, z, step=current_step)
                    output_mask = tf.tile(tf.expand_dims(tf.ones(tf.shape(z)) - z, axis=-1), [1,1,self.vocab_size])
                    final_softmax = self.output_layer(outputs) * output_mask
                    loss = loss_fn(y, final_softmax, z)
            
            num_segments += 1
            loss_values.append(loss)
            all_outputs.append(final_softmax)
            ## -----------l--------------------------------------------------------
            trainable_variables = self.collect_trainable_variables()
            grads_and_vars = zip(tape.gradient(loss, trainable_variables), trainable_variables)
            grads_and_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
            self.optimizer.apply_gradients(grads_and_vars)
            ## -------------------------------------------------------------------
            tf.summary.scalar('Cross Entropy Loss', loss, step=current_step)
        ## -------------------------------------------------------------------
        total_loss = tf.reduce_mean(loss_values)
        return total_loss, tf.concat(all_outputs, axis=1)
        ## -------------------------------------------------------------------

        
    ## -------------------------------------------------------------------
    def collect_trainable_variables(self):
        variables = [self.trainable_variables]
        if self.create_encoder:
            variables.append(self.encoder_stack.trainable_variables)
        if self.create_decoder:
            variables.append(self.decoder_stack.trainable_variables)
        variables = [item for sublist in variables for item in sublist]
        return variables


    ## -------------------------------------------------------------------
    ## TASK - MUSIC GENERATION (DECODER ONLY)
    ## -------------------------------------------------------------------
    def music_generation_preprocess(self, tracks, inp_key, out_key):
        inputs = self.create_musical_segments(
            tf.sparse.to_dense(tracks[inp_key])[:,:-1], inp_key
        )
        if (bool(out_key)):
            outputs = self.create_musical_segments(
                tf.sparse.to_dense(tracks[out_key])[:,1:], out_key
            )
        full_inputs = tf.concat(inputs, axis=-1)
        segment_length = tf.shape(full_inputs)[1]
        num_splits = segment_length / self.max_sequence_length
        padding_mask = tf.squeeze(create_padding_mask(full_inputs))
        padding_mask = tf.split(
            padding_mask,
            num_or_size_splits=int(num_splits),
            axis=1
        )
        return inputs, outputs, padding_mask
    ## -------------------------------------------------------------------
    def music_generation(self, inputs, ouputs, pe, p_mask, step=0):
        tf.summary.trace_on(graph=True)
        decoder_output = self.run_decoder_stack(
            inputs, pe, self.memory_length, p_mask,
            encoder_outputs=None, training=None
        )
        tf.summary.trace_export(name='MIDI Transformer', step=step)
        return decoder_output
    ## -------------------------------------------------------------------
    def music_generation_loss(self, targets, outputs, mask):
        loss = self.cross_entropy(targets, outputs)
        mask = tf.ones(tf.shape(mask)) - mask
        loss *= mask
        return tf.reduce_sum(loss)/tf.reduce_sum(mask)
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    ## TASK - MUSIC INPAINTING (ENCODER ONLY)
    ## -------------------------------------------------------------------
    def music_inpainting_preprocess(self, tracks, inp_key, out_key):
        inputs = self.create_musical_segments(tracks[inp_key], inp_key)
        outputs = self.create_musical_segments(tracks[out_key], out_key)
        padding_mask = None
        train_mask = None
        return inputs, outputs, train_mask, padding_mask
    ## -------------------------------------------------------------------
    def music_inpainting(self, inputs, outputs, pe, t_mask, p_mask):
        return self.run_encoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask, training=True
        )
    ## -------------------------------------------------------------------
    def music_inpainting_loss(self, inputs, outputs):
        pass
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    ## TASK - MUSIC TRANSLATION (ENCODER + DECODER)
    ## -------------------------------------------------------------------
    def musical_translation_preprocess(self, song, input_key, output_key):
        inputs = self.create_musical_segments(tracks[inp_key], inp_key)
        outputs = self.create_musical_segments(tracks[out_key], out_key)
        padding_mask = None
        train_mask = None
        return inputs, outputs, train_mask, padding_mask
    ## -------------------------------------------------------------------
    def musical_translation(self, inputs, ouputs, pe, t_mask, p_mask):
        encoder_outputs = self.run_encoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask, training=True
        )
        return self.run_decoder_stack(
            inputs, pe, self.memory_length, t_mask, p_mask,
            encoder_outputs=encoder_outputs, training=True
        )
    ## -------------------------------------------------------------------
    def music_translation_loss(self, inputs, outputs):
        pass
    ## -------------------------------------------------------------------


    # ------------------------------------------------------------------------------        
    def create_musical_segments(self, inputs, key, split=True):
        # ------------------------------------------------------------------------------
        inp_pad_size = 0
        segment_length = tf.shape(inputs)[1]
        if (segment_length % self.max_sequence_length):
            inp_pad_size = int(segment_length / self.max_sequence_length)
            inp_pad_size = ((inp_pad_size + 1) * self.max_sequence_length) - segment_length
            inputs = tf.pad(inputs, [[0, 0],[0, inp_pad_size]])
        # ------------------------------------------------------------------------------        
        if split:
            num_splits = math.ceil(segment_length / self.max_sequence_length)
            return tf.split(
                inputs,
                num_or_size_splits=num_splits, 
                axis=1
            )
        else:
            return inputs
        # ------------------------------------------------------------------------------        
    # ------------------------------------------------------------------------------        


    ## -------------------------------------------------------------------
    def call(self, inputs, targets, positional_encoding, memory_length):
        return self.run_step(inputs, targets, positional_encoding, memory_length)
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def reset_model_state(self):
        if self.create_encoder:
            self.encoder_stack.reset_states()
        if self.create_decoder:
            self.decoder_stack.reset_states()
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def run_encoder_stack(
        self, inputs, positional_encoding, memory_length, padding_mask,
        training=True
    ):
        ## -------------------------------------------------------------------
        embeddings = self.encoder_embedding(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.enc_dropout(embeddings, training=training)

        return self.encoder_stack(
            embeddings,                 # Embedding of MIDI NoteSequence
            memory_length,              # Length of previous memories
            positional_encoding,        # Sinusoidal positional encoding
            t_mask,                     # Task related mask (Look ahead mask or Masked-LM mask)
            padding_mask,               # Padding mask for input
            training,                   # Mode of operation
        )
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    @tf.function
    def run_decoder_stack(
        self, inputs, positional_encoding, memory_length, padding_mask, 
        encoder_outputs=None, training=None
    ):
        ## -------------------------------------------------------------------
        embeddings = self.decoder_embedding(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.dec_dropout(embeddings, training=training)

        decoder_output = self.decoder_stack(
            embeddings,                 # Embedding of MIDI NoteSequence
            memory_length,              # Length of memory of previous segments
            positional_encoding,        # Sinusoidal positional encoding
            padding_mask,               # Padding mask for input
            encoder_outputs,            # Outputs from encoder if any
            training,                   # Mode of operation
        )
        return decoder_output
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def create_or_restore_checkpoint(self, name, model, optimizer):
        ## -------------------------------------------------------------------
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        ckpt_path = os.path.join(self.model_path, 'ckpt', name)
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=3)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        ## -------------------------------------------------------------------
        if ckpt_manager.latest_checkpoint:
            print("Restored" + name.upper() + "from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing " + name.upper() + " from scratch.")
        return ckpt, ckpt_manager
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------


    ## -------------------------------------------------------------------
    def save_model_checkpoint(self):
        ## -------------------------------------------------------------------
        if self.create_encoder:
            encoder_save_path = self.enc_ckpt_manager.save()
            step = int(self.enc_ckpt.step)
            print("Saved checkpoint for step {}: {}".format(step, encoder_save_path))

        if self.create_decoder:
            decoder_save_path = self.dec_ckpt_manager.save()
            step = int(self.dec_ckpt.step)
            print("Saved checkpoint for step {}: {}".format(step, decoder_save_path))
        ## -------------------------------------------------------------------
    ## -------------------------------------------------------------------
