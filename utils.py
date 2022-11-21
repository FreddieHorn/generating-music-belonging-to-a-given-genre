import numpy as np 
import tensorflow as tf
import pretty_midi
import glob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_token(unknown_tag_id, logits, k):
    logits, indices = tf.math.top_k(logits, k= k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = np.asarray(logits).astype("float32")
    if(unknown_tag_id in indices):
        unk_tag_position = np.where(indices == unknown_tag_id)[0].item()
        indices = np.delete(indices, unk_tag_position)
        preds = np.delete(preds, unk_tag_position)
    preds = softmax(preds)
    choice=np.random.choice(indices, p=preds)
    #alternative choice (unstable)
    #choice=indices[np.argmax(preds)]
    return choice

def convertToRoll(int_to_combi, binarizer, seq_list):
    seq_list = [int_to_combi[i] for i in seq_list]
    roll = binarizer.transform(seq_list)
    return roll

def piano_roll_to_pretty_midi(piano_roll_in, fs, program=0, velocity = 64):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    piano_roll = np.where(piano_roll_in == 1, 64, 0)
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI(initial_tempo=100.0)
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    print(piano_roll.shape)
    
    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def save_midi(name, path, out_piano_roll):
    bpm = 150
    fs = 1/((60/bpm)/4)
    mid_out = piano_roll_to_pretty_midi(out_piano_roll.T, fs=fs)
    midi_out_path = path+f"{name}.mid"
    if midi_out_path is not None:
            mid_out.write(midi_out_path)


def get_midi_files():
    directory = "POP9/**"
    for filename in glob.iglob(directory, recursive=True):
        print(filename)

if __name__ == '__main__':
    get_midi_files()