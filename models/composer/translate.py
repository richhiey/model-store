import os
import tensorflow as tf

from .helpers.layers import TransformerEncoderStack, TransformerDecoderStack
from .helpers.utils import Transformer_LR_Schedule, create_segments


class TransformerTranslator(tf.keras.Model):

    def __init__(self, configs, rate=0.1):
        super(TransformerTranslator, self).__init__()

        self.encoder = TransformerEncoderStack(
            configs['num_layers'], configs['d_model'], 
            configs['num_heads'], configs['dff'], 
            configs['input_vocab_size'], configs['pe_input'], rate
        )
        self.decoder = TransformerDecoderStack(
            configs['num_layers'], configs['d_model'], 
            configs['num_heads'], configs['dff'], 
            configs['target_vocab_size'], configs['pe_target'], rate
        )
        self.final_layer = tf.keras.layers.Dense(configs['target_vocab_size'])
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        self.optimizer = tf.keras.optimizers.Adam(Transformer_LR_Schedule(configs['d_model']))
        self.ckpt = tf.train.Checkpoint(
            step = tf.Variable(1),
            optimizer = self.optimizer,
            model = self
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(configs['model_path'], 'ckpt'),
            max_to_keep = 3
        )
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")


    def __call_model__(self, inp, tar, training, enc_padding_mask, 
                    look_ahead_mask, dec_padding_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def call(self, inp, tar, enc_padding_mask, look_ahead_mask,
        dec_padding_mask, training=True):
        return self.__call_model__(
            inp, tar, training, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask
        )


    def loss_function(self, targets, predictions):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_fn(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



    def train_step(self, segment):
        inputs = sequence[:, :-1]
        targets = sequence[:, 1:]

        with tf.GradientTape() as tape:
            outputs = self.__call_model__(
                inputs, targets,
                enc_padding_mask, dec_padding_mask,
                look_ahead_mask,
            )
            loss = self.loss_function(outputs, targets)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, outputs

    def train(self, dataset, configs):
        for epoch in range(configs['num_epochs']):
            for i, sequence in enumerate(dataset):

                losses = []
                curr_step = int(self.ckpt.step)

                segments = create_segments(sequence)
                
                for segment in segments:
                    segment = tf.squeeze(segment)
                    loss_val, outputs = self.train_step(segment)
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