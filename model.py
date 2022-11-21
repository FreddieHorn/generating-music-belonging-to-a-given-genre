import numpy as np
from transformer import TransformerBlock, MultiHeadSelfAttention, TokenAndPositionEmbedding, InputDataGenerator, GeneratorCallback
from utils import get_token, softmax, convertToRoll, piano_roll_to_pretty_midi, save_midi
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

class NetModel:
    def __init__(self, maxlen, embed_dim, num_heads, feed_forward_dim, vocab_size) -> None:
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = feed_forward_dim
        self.vocab_size = vocab_size
    def create_model(self):
        inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        x = embedding_layer(inputs)
        transformer_block1 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, dropout_rate = 0.25)
        transformer_block2 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, dropout_rate = 0.25)
        transformer_block3 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, dropout_rate = 0.25)
        transformer_block4 = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, dropout_rate = 0.25)
        x = transformer_block1(x)
        x = transformer_block2(x)
        x = transformer_block3(x)
        x = transformer_block4(x)
        outputs = Dense(self.vocab_size)(x)
        model = Model(inputs=inputs, outputs=[outputs,x])
        fun_loss = SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer="adam", loss=[fun_loss,None])
        #print(model.summary()) 
        self.model = model

        return model

    def plot_model_pic(self):
        plot_model(self.model, to_file='model2.png', show_shapes=True, show_layer_names=True)

    def train(self, all_song_tokenised, batch_size, epochs, sequence_length, path):
        train_loss = []
        val_loss = []
        #90% train, 10% val
        train_data = InputDataGenerator(all_song_tokenised[:0.9*len(all_song_tokenised)], batch_size, sequence_length)
        val_data = InputDataGenerator(all_song_tokenised[0.9*len(all_song_tokenised):], batch_size, sequence_length)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )

        history = self.model.fit(x = train_data,
                    callbacks = [checkpoint],                    
                   epochs = epochs,
                   verbose = 1,
                   validation_data = val_data)

        train_loss += history.history['loss']
        val_loss += history.history['loss']

        plt.plot(train_loss)
        plt.title('model train')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        try:
            plt.savefig(f'{epochs}_loss.png')
        except:
            print("Something went wrong during saving a matplotlib image")

class MusicGenerator():
    def __init__(self, starting_seed_len, no_of_tones_to_gen, num_of_songs, mlb, unk_tag_idx) -> None:
        self.starting_seed_len = starting_seed_len
        self.num_note_to_gen = no_of_tones_to_gen
        self.NUM_SONGS = num_of_songs
        self.mlb = mlb
        self.unk_tag_id = unk_tag_idx

    def choose_starting_tones(self, all_songs_tokenised, seq_length):
        self.seq_length = seq_length
        song_idx = random.randint(0,len(all_songs_tokenised)-1)
        seq_start_at = random.randint(0,len(all_songs_tokenised[song_idx])-seq_length)   
        self.start_tokens = all_songs_tokenised[song_idx][seq_start_at:seq_start_at + self.starting_seed_len].tolist()

        while (self.start_tokens == [()]*seq_length):
            print("Got all zeros, rerolling")
            song_idx = random.randint(0,len(all_songs_tokenised)-1)
            seq_start_at = random.randint(0,len(all_songs_tokenised[song_idx])-seq_length)   
            self.start_tokens = all_songs_tokenised[song_idx][seq_start_at:seq_start_at + seq_length].tolist()
            



        self.tokens_generated = []
        self.num_tokens_generated = 0


    def generate_tones(self, maxlen, model, int_to_combi,all_songs_tokenised, seq_length, name_of_songs = "song_"):
        for i in range(self.NUM_SONGS):
            self.choose_starting_tones(all_songs_tokenised, seq_length)
            while self.num_tokens_generated <= self.num_note_to_gen:

                x = self.start_tokens[-self.seq_length:]
                pad_len = maxlen - len(self.start_tokens)
                sample_index = -1
                if pad_len > 0:
                    x = self.start_tokens + [0] * pad_len
                    sample_index = len(self.start_tokens) - 1
                x = np.array([x])
                y, logs = model.predict(x)
                predicted_token = get_token(self.unk_tag_id, y[0][sample_index], 10)
                self.tokens_generated.append(predicted_token)
                self.start_tokens.append(predicted_token)
                self.num_tokens_generated = len(self.tokens_generated)
                print(f"generated {self.num_tokens_generated} notes || SONG {i} / {self.NUM_SONGS}")
            piano_roll = convertToRoll(int_to_combi, self.mlb, self.start_tokens)
            save_midi(f"{name_of_songs}nr{i}", "./generated", piano_roll)
        