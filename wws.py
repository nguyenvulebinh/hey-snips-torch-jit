import torch
import torchaudio.compliance.kaldi as kaldi
import pyaudio
import queue
import numpy as np
import threading
import time

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze()
    return sound

def compute_mfcc(
    waveform,
    num_ceps=80,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
    sample_rate=16000
):
    waveform = waveform * (1 << 15)
    mat = kaldi.mfcc(
        waveform,
        num_ceps=num_ceps,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    )
    return mat


class RealtimeDecoder():

    def __init__(self,
        model_jit
    ) -> None:
        self.model_jit = model_jit
        self.SAMPLE_RATE = 16000
        self.cache_output = {
            "wavchunks": [],
            "prob": torch.zeros(0, 0, 0, dtype=torch.float)
        }

        self.continue_recording = threading.Event()
        self.frame_duration_ms = 500
        self.audio_queue = queue.SimpleQueue()

    def start_recording(self, wait_enter_to_stop=True):
        def stop():
            input("Press Enter to stop the recording:\n\n")
            self.continue_recording.set()
        def record():
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=int(self.SAMPLE_RATE / 10))
            while not self.continue_recording.is_set():
                audio_chunk = stream.read(int(self.SAMPLE_RATE * self.frame_duration_ms / 1000.0), exception_on_overflow = False)
                audio_int16 = np.frombuffer(audio_chunk, np.int16)
                audio_float32 = int2float(audio_int16)
                waveform = torch.from_numpy(audio_float32)
                self.audio_queue.put(waveform)
            print("Finish record")
            stream.close()
        if wait_enter_to_stop:
            stop_listener_thread = threading.Thread(target=stop, daemon=False)
        else:
            stop_listener_thread = None
        recording_thread = threading.Thread(target=record, daemon=False)
        return stop_listener_thread, recording_thread

    def start_decoding(self):
        def decode():
            count_wav = 0
            while not self.continue_recording.is_set():
                if self.audio_queue.qsize() > 0:
                    currunt_wavform = self.audio_queue.get()
                    self.cache_output['wavchunks'].append(currunt_wavform)
                    self.cache_output['wavchunks'] = self.cache_output['wavchunks'][-2:]
                    wavform = torch.cat(self.cache_output['wavchunks'], dim=-1)
                    feat = compute_mfcc(waveform=wavform.unsqueeze(0), sample_rate=self.SAMPLE_RATE)[-49:]
                    speech = feat.unsqueeze(0)
                    prob = self.cache_output['prob']
                    feats, prob = self.model_jit.forward(speech, prob)
                    self.cache_output['prob'] = prob
                    score = feats.max().detach().numpy().tolist()
                    if score > 0.1:
                        print("Wake word detected. Score: {:.4f}".format(score))
                        count_wav += 1
                    else:
                        print('.')
                else:
                    time.sleep(0.01)
            print("Decode thread finish")
        decode_thread = threading.Thread(target=decode, daemon=False)
        return decode_thread

if __name__ == "__main__":
    print("Model loading....")
    model = torch.jit.load('model.zip').eval()
    print("Model loaded....")    

    obj_decode = RealtimeDecoder(model)
    recording_threads = obj_decode.start_recording()
    decode_thread = obj_decode.start_decoding()
    for thread in recording_threads:
        if thread is not None:
            thread.start()
    decode_thread.start()
    for thread in recording_threads:
        if thread is not None:
            thread.join()
    decode_thread.join()