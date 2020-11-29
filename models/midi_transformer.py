## MIDI Transformer on Pop909 dataset

import tensorflow as tf
from helpers.layers import TransformerEncoder, TransformerDecoder

class MIDITransformer(tf.keras.Model):
## -------------------------------------------------------------------
    def __init__(self, config_path, model_path):
        ## -------------------------------------------------------------------
        self.model_path = model_path
        if os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        ## -------------------------------------------------------------------
        self.config_path = config_path
        with open(self.config_path) as json_file: 
            self.configs = json.load(self.config_path)
        ## -------------------------------------------------------------------
        self.midi_encoder = TransformerXLEncoderStack(self.configs['encoder'])
        self.midi_decoder = TransformerXLDecoderStack(self.configs['encoder'])
        self.model = self.__create_model__()
        ## -------------------------------------------------------------------
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
        saved_model_dir = os.path.join(self.model_path, 'MIDI_Transformer')
        end_learning_rate = 0.00001
        decay_steps = 100000.0
        decay_rate = 0.
        learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(
          initial_learning_rate, decay_steps, end_learning_rate, power=3
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate_fn)
        ## -------------------------------------------------------------------


    def __create_model__(self):
        input_midi = tf.keras.layers.Input((256, 256))
        midi_embedding = self.midi_encoder(input_midi)
        output_midi = self.midi_decoder(midi_embedding)
        model = tf.keras.Model(inputs = input_midi, output = output_midi)
        return model
