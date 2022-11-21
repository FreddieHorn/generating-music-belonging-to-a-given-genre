from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from model import NetModel, MusicGenerator
from processing_data import DataManager

#___________________CONFIG PARAMS______________
EXPORT_DATA_JAZZ  = "./exported_jazz_v2/"
EXPORT_DATA_POP  = "./exported_pop_v2/"
EXPORT_DATA_POP_TEST  = "./exported_poptest/"
mlb = MultiLabelBinarizer()
mlb.fit([np.arange(128).tolist()])

batch_size= 32
sequence_length = 600
generate_sample_every_ep = 100

maxlen = sequence_length  
embed_dim = 128  
num_heads = 4  
feed_forward_dim = 128  
MAX_VOCAB_SIZE = 73911 #depends what do u want to generate

unk_tag_str = '<UNK>'
unk_tag_idx = 0
pad_tag_str = ''
pad_tag_idx = 1

#vocab_size_jazz = 73911
#vocab_size_pop = 34973 

if __name__ == '__main__':
# 
    #_______PREPROCESSING DATA________
    data_manager = DataManager(MAX_VOCAB_SIZE, unk_tag_idx, unk_tag_str, pad_tag_idx, pad_tag_str)
    vocab, all_songs_tokenised, int_to_combi = data_manager.process_song_data(mlb, EXPORT_DATA_JAZZ)
    vocab_size = len(vocab)
    #_____________CREATING MODEL_______________
    epochs= 1500
    save_path = f"./transformer_output/<FILENAME>"
    load_path = f"./transformer_output/jazzOnly.hdf5"
    model_creator = NetModel(maxlen, embed_dim, num_heads, feed_forward_dim, vocab_size)
    model_creator.create_model()
    #EITHER
    model_creator.model.load_weights(load_path)
    #OR
    #model_creator.train(all_songs_tokenised, batch_size, epochs, sequence_length, save_path)

    #_____________GENERATING MUSIC________________
    starting_seed_len = 300
    number_tones_to_gen = 1000
    num_songs_to_gen = 10
    music_generator = MusicGenerator(starting_seed_len, number_tones_to_gen, num_songs_to_gen, mlb, unk_tag_idx)
    music_generator.generate_tones(maxlen, model_creator.model, int_to_combi, all_songs_tokenised, sequence_length , "jazz-")


    # # #____________PROCESSING POP SONGS______________
    vocab, all_songs_tokenised, int_to_combi = data_manager.process_song_data(mlb, EXPORT_DATA_POP)
    vocab_size = len(vocab)
    
    # # #____________TRANSFER LEARNING_____________
    epochs_pop = 750 #LESS EPOCHS BECAUSE MODEL IS PRE_TRAINED
    save_path = f"./transformer_output/<FILENAME>"
    load_path = f"./transformer_output/popTransfer.hdf5"
    model_creator.model.load_weights(load_path)
    #model_creator.train(all_songs_tokenised, batch_size, epochs_pop, sequence_length, save_path)
    music_generator.generate_tones(maxlen, model_creator.model, int_to_combi, all_songs_tokenised, sequence_length, f"pop-")
