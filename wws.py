import torch
import torchaudio.compliance.kaldi as kaldi
import pyaudio
import queue
import numpy as np
import threading
import time
import math

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

def compute_fbank(waveform,
                  feature_type='fbank',
                  num_mel_bins=40,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0,
                  sample_rate=16000):
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                        num_mel_bins=num_mel_bins,
                        frame_length=frame_length,
                        frame_shift=frame_shift,
                        dither=dither,
                        energy_floor=0.0,
                        sample_frequency=sample_rate)
    return mat

class RealtimeDecoder():

    def __init__(self,
        model_jit,
        inference_duration_s: float = 1.5
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
        self.inference_duration_s = inference_duration_s
        
        # fbank params from compute_fbank
        frame_length_ms = 25
        frame_shift_ms = 10
        
        # Calculate number of frames for inference window
        duration_ms = self.inference_duration_s * 1000
        self.num_frames_for_inference = int(((duration_ms - frame_length_ms) / frame_shift_ms) + 1)
        
        # Calculate number of chunks for buffer. Buffer must be >= inference window.
        self.num_chunks_to_keep = math.ceil(self.inference_duration_s * 1000 / self.frame_duration_ms)

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
                self.audio_queue.put((waveform, time.time()))
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
            while not self.continue_recording.is_set():
                if self.audio_queue.qsize() > 0:
                    currunt_wavform, timestamp = self.audio_queue.get()
                    self.cache_output['wavchunks'].append(currunt_wavform)
                    self.cache_output['wavchunks'] = self.cache_output['wavchunks'][-self.num_chunks_to_keep:]
                    wavform = torch.cat(self.cache_output['wavchunks'], dim=-1)

                    # 1. Feature extraction latency
                    feature_extraction_start = time.time()
                    feat = compute_fbank(waveform=wavform.unsqueeze(0), sample_rate=self.SAMPLE_RATE)[-self.num_frames_for_inference:]
                    feature_extraction_end = time.time()
                    feature_extraction_latency = feature_extraction_end - feature_extraction_start

                    speech = feat.unsqueeze(0)
                    prob = self.cache_output['prob']

                    # 2. Model inference latency
                    inference_start = time.time()
                    feats, prob = self.model_jit.forward(speech, prob)
                    inference_end = time.time()
                    inference_latency = inference_end - inference_start

                    self.cache_output['prob'] = prob
                    score = feats.max().detach().numpy().tolist()
                    
                    # 3. Total latency
                    total_latency = time.time() - timestamp

                    if score > 0.5:
                        print("Wake word detected. Score: {:.4f}".format(score))
                    else:
                        print(f"Latency - Total: {total_latency:.4f}s, Feature Extraction: {feature_extraction_latency:.4f}s, Inference: {inference_latency:.4f}s")
                else:
                    time.sleep(0.01)
            print("Decode thread finish")
        decode_thread = threading.Thread(target=decode, daemon=False)
        return decode_thread

if __name__ == "__main__":
    print("Model loading....")
    model = torch.jit.load('model.zip').eval()
    print("Model loaded....")    

    obj_decode = RealtimeDecoder(model, inference_duration_s=1.0)
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