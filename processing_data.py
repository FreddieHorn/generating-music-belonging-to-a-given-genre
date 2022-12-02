import numpy as np
import pretty_midi
from music21 import *
import glob



DATA_PATH = "./jazz/"
EXPORT_DATA_JAZZ  = "./exported_jazz/"

DATA_PATH_POP = "./pop/"

EXPORT_DATA_POP = "./exported_pop/"
class MIDIparser:
    def __init__(self) -> None:
        self.extract_to_npy(DATA_PATH, EXPORT_DATA_JAZZ)
        self.extract_to_npy(DATA_PATH_POP, EXPORT_DATA_POP)

    def preprocess_midi(self,path, offset_by, bpm):
        parsed_midi = pretty_midi.PrettyMIDI(midi_file=path)
        filtered_instruments = [inst for inst in parsed_midi.instruments if ((len(inst.notes) > 0) and
                                                        (inst.is_drum == False) and
                                                        (inst.program < 8)
                                                    )]

        piano = filtered_instruments[np.argmax([len(inst.notes) for inst in filtered_instruments])]
                
        start_time = piano.notes[0].start
        end_time = piano.get_end_time()
        
        quater_note_len = 60/bpm
        thrity_two_note = 8
        fs = 1/(quater_note_len/thrity_two_note)
        
        piano_roll_matrix = piano.get_piano_roll(fs = fs, times = np.arange(start_time, end_time,1./fs))
        piano_roll_matrix = np.roll(piano_roll_matrix, -offset_by)
        output_pianoroll = np.where(piano_roll_matrix > 0, 1,0)

        return output_pianoroll.T

    def extract_bpm_and_offset(self, path):
        mid = converter.parse(path)
        all_instrument_parts = instrument.partitionByInstrument(mid)
        piano_part = all_instrument_parts.parts[0]

        for i in piano_part:
            if isinstance(i, tempo.MetronomeMark):
                bpm = i.getQuarterBPM()
                break
        # try:
        #     key = piano_part.keySignature
        # except:
        #     print(f"Error while finding key signature for song")
        # try:
        #     key_in_major = key.asKey(mode='major')
        #     offset_by = key_in_major.tonic.pitchClass
        # except:
        #     print("alternative offset")
        offset_by = piano_part.offset
        return offset_by, bpm

    def process_piano_roll(self, piano_roll, max_consecutive = 64):
#     This function is to remove consecutive notes that last for more than roughtly 2 secs
        prev = np.random.rand(128)
        count = 0
        remove_idxs = []
        remove_slice = []
        for idx, piano_slice in enumerate(piano_roll):
            if(np.array_equal(prev, piano_slice)):
                count+=1
                if (count > max_consecutive):
                    remove_idxs.append(idx)
                    if (str(piano_slice) not in remove_slice):
                        remove_slice.append(str(piano_slice))
            else:
                count = 0
            prev = piano_slice
        out_piano_roll = np.delete(piano_roll, remove_idxs, axis=0)
        return out_piano_roll

    def extract_to_npy(self, data_path, export_path):
        failed_list = []

        for temp in glob.glob(data_path + "*.mid"):
            try:
                print(temp)
                offset_by, bpm = self.extract_bpm_and_offset(temp)
                piano_roll = self.preprocess_midi(temp, offset_by, bpm)
                #piano_roll = self.process_piano_roll(piano_roll)
                name  = temp.split("\\")[-1].split(".")[0]
                out_name = f"{export_path}encoded_{name}.npy"
                np.save(out_name, piano_roll)
                print(f"saved {out_name}")

            except Exception as e:
                print(f"Failed to preprocess {temp}")
                print(e)
                failed_list.append(temp)
                continue

class DataManager:
    def __init__(self, max_vocab_size, unk_tag_idx, unk_tag_str, pad_tag_idx, pad_tag_str) -> None:
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.unk_tag_idx = unk_tag_idx,  
        self.unk_tag_str = unk_tag_str,
        self.pad_tag_idx = pad_tag_idx,
        self.pad_tag_str  = pad_tag_str 

    def process_song_data(self, mlb, export_data):
        all_songs = []
        all_songs_np = np.empty((0,128), np.int8)
        for temp in glob.glob(export_data + "*.npy"):
            encoded_data = np.load(temp).astype(np.int8)
            all_songs.append(encoded_data)
            all_songs_np = np.append(all_songs_np, encoded_data, axis=0)

        unique_np, counts = np.unique(all_songs_np, axis=0,return_counts=True)
        unique_note_intergerized = np.array(mlb.inverse_transform(unique_np))
        count_sort_ind = np.argsort(-counts)

        vocab = unique_note_intergerized[count_sort_ind][:self.MAX_VOCAB_SIZE-2].tolist()
        top_counts = counts[count_sort_ind][:self.MAX_VOCAB_SIZE-1].tolist()

        vocab.sort(key=len)
        vocab.insert(self.unk_tag_idx[0], self.unk_tag_str[0])
        vocab.insert(self.pad_tag_idx[0], self.pad_tag_str)


        combi_to_int = dict((combi, number) for number, combi in enumerate(vocab))
        int_to_combi = dict((number, combi) for number, combi in enumerate(vocab))

        all_songs_tokenised = []
        for idx, song in enumerate(all_songs):
            song = mlb.inverse_transform(song)
            song = [combi_to_int[tup] if tup in vocab else self.unk_tag_idx[0] for tup in song]
            all_songs_tokenised.append(np.array(song))
        return vocab, all_songs_tokenised, int_to_combi
if __name__ == "__main__":
    MIDIparser()