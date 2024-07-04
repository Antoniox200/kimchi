# import tkinter as tk
# import pyaudio
# import wave
# import threading
# import pygame
# import io
# import numpy as np
# from pydub import AudioSegment
# from openai import OpenAI
# import webrtcvad
# import collections
# import time

# client = OpenAI(api_key="sk-proj-oQpqMVR2jfkrJuWhzhjfT3BlbkFJd41WdvJe6qzW2SXO2dT9")

# class ConversationContext:
#     def __init__(self):
#         self.messages = []

#     def add_message(self, role, content):
#         self.messages.append({"role": role, "content": content})
    
#     def get_messages(self):
#         return self.messages

# class VADInterruptibleAIAssistant:
#     def __init__(self, master):
#         self.master = master
#         master.title("VAD-enabled Interruptible AI Assistant")

#         self.conversation_context = ConversationContext()
#         self.is_listening = False
#         self.is_speaking = False
#         self.last_speech_time = time.time()
#         self.debounce_time = 1  # 1000ms debounce time
#         self.interrupted_speech = False
#         self.min_audio_length = 0.1  # 100ms minimum audio length for transcription
#         self.energy_threshold = 500  # Adjust based on your audio input
#         self.silence_threshold = 0.1  # 10% of frames can be silence

#         self.text_output = tk.Text(master, height=20, width=50)
#         self.text_output.pack()

#         self.p = pyaudio.PyAudio()
#         pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)

#         self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
#         self.buffer_queue = collections.deque(maxlen=50)  # 500ms audio buffer
#         self.speech_buffer = []

#         self.short_interrupt_buffer = collections.deque(maxlen=100)  # 1000ms buffer for short interruptions
#         self.tts_paused = False
#         self.pause_event = threading.Event()
        
#         self.potential_interruption_buffer = []
#         self.in_potential_interruption = False
#         self.potential_interruption_timer = None

#         # TTS queue and lock
#         self.tts_queue = collections.deque()
#         self.tts_lock = threading.Lock()
#         self.tts_thread = None
#         self.tts_event = threading.Event()

#         self.start_listening()
        
#         # Start TTS playback thread
#         self.tts_thread = threading.Thread(target=self.tts_player)
#         self.tts_thread.daemon = True
#         self.tts_thread.start()

#     def start_listening(self):
#         self.is_listening = True
#         self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320, stream_callback=self.audio_callback)
#         self.stream.start_stream()

#     def audio_callback(self, in_data, frame_count, time_info, status):
#         frame_duration = 10  # frame duration in ms
#         frame_size = int(16000 * frame_duration / 1000 * 2)  # number of bytes per frame
#         frames = [in_data[i:i + frame_size] for i in range(0, len(in_data), frame_size)]

#         is_speech_detected = False
#         for frame in frames:
#             if len(frame) == frame_size:
#                 is_speech = self.vad.is_speech(frame, 16000)
#                 if is_speech:
#                     is_speech_detected = True
#                     self.last_speech_time = time.time()
#                     self.speech_buffer.append(frame)
#                     self.buffer_queue.append(frame)
#                     if self.is_speaking:
#                         self.short_interrupt_buffer.append(True)
#                         if not self.in_potential_interruption:
#                             self.start_potential_interruption()
#                         self.potential_interruption_buffer.append(frame)
#                         if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and all(self.short_interrupt_buffer):
#                             self.confirm_interruption()
#                     else:
#                         self.short_interrupt_buffer.clear()
#                 else:
#                     self.buffer_queue.append(frame)
#                     if self.is_speaking:
#                         self.short_interrupt_buffer.append(False)
#                         if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and not any(self.short_interrupt_buffer):
#                             self.end_potential_interruption()

#         if not is_speech_detected and len(self.buffer_queue) == self.buffer_queue.maxlen:
#             current_time = time.time()
#             if current_time - self.last_speech_time > self.debounce_time and not self.in_potential_interruption:
#                 self.process_audio(self.speech_buffer)
#                 self.speech_buffer.clear()
#                 self.buffer_queue.clear()

#         return (in_data, pyaudio.paContinue)

#     def start_potential_interruption(self):
#         if not self.in_potential_interruption:
#             print("Starting potential interruption.")
#             self.in_potential_interruption = True
#             self.pause_tts()
#             self.potential_interruption_timer = threading.Timer(2.0, self.end_potential_interruption)
#             self.potential_interruption_timer.start()

#     def confirm_interruption(self):
#         if self.in_potential_interruption:
#             print("Confirming interruption.")
#             self.interrupt_speech()
#             self.in_potential_interruption = False
#             if self.potential_interruption_timer:
#                 self.potential_interruption_timer.cancel()
#             self.potential_interruption_timer = None
#             self.process_audio(self.potential_interruption_buffer + self.speech_buffer)
#             self.potential_interruption_buffer.clear()
#             self.speech_buffer.clear()

#     def end_potential_interruption(self):
#         if self.in_potential_interruption:
#             print("Ending potential interruption.")
#             self.in_potential_interruption = False
#             self.resume_tts()
#             if self.potential_interruption_timer:
#                 self.potential_interruption_timer.cancel()
#             self.potential_interruption_timer = None
#             self.potential_interruption_buffer.clear()

#     def interrupt_speech(self):
#         if self.is_speaking:
#             print("Interrupting speech.")
#             pygame.mixer.stop()
#             self.is_speaking = False
#             self.interrupted_speech = True
#             self.tts_paused = False
#             self.text_output.insert(tk.END, "Speech interrupted.\n")
#             with self.tts_lock:
#                 self.tts_queue.clear()

#     def pause_tts(self):
#         if self.is_speaking and not self.tts_paused:
#             print("Pausing speech.")
#             pygame.mixer.pause()
#             self.tts_paused = True
#             self.pause_event.set()

#     def resume_tts(self):
#         if self.is_speaking and self.tts_paused:
#             print("Resuming speech.")
#             pygame.mixer.unpause()
#             self.tts_paused = False
#             self.pause_event.clear()

#     def tts_player(self):
#         while True:
#             self.tts_event.wait()
#             with self.tts_lock:
#                 if self.tts_queue:
#                     next_tts = self.tts_queue.popleft()
#                     self.play_audio(next_tts['sound'], next_tts['duration'])
#                 else:
#                     self.tts_event.clear()

#     def play_audio(self, sound, duration):
#         self.is_speaking = True
#         sound.play()
#         start_time = time.time()
#         elapsed_time = 0
#         while elapsed_time < duration and self.is_speaking:
#             if self.pause_event.is_set():
#                 pygame.mixer.pause()
#                 self.pause_event.wait()
#                 pygame.mixer.unpause()
#                 start_time = time.time() - elapsed_time
#             elapsed_time = time.time() - start_time
#             time.sleep(0.1)
#         self.stop_speaking()

#     def stop_speaking(self):
#         pygame.mixer.stop()
#         self.is_speaking = False
#         self.tts_event.set()  # Signal that we're ready for the next TTS

#     def is_silence(self, audio_segment):
#         return np.sqrt(np.mean(np.square(audio_segment))) < self.energy_threshold

#     def contains_speech(self, audio_frames):
#         silence_count = sum(1 for frame in audio_frames if self.is_silence(np.frombuffer(frame, dtype=np.int16)))
#         return (len(audio_frames) - silence_count) / len(audio_frames) > self.silence_threshold

#     def process_audio(self, audio_frames):
#         if not audio_frames:
#             return

#         self.is_listening = False
#         print("Processing audio frames.")

#         audio_length = len(audio_frames) * 10 / 1000  # in seconds
#         if audio_length < self.min_audio_length:
#             print("Audio too short for transcription: {audio_length} seconds.")
#             self.is_listening = True
#             return
        
#         # Check if audio contains speech
#         if not self.contains_speech(audio_frames):
#             print("Audio doesn't contain enough speech.")
#             self.is_listening = True
#             return

#         # Write audio frames to WAV file
#         wf = wave.open("temp.wav", "wb")
#         wf.setnchannels(1)
#         wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(16000)
#         wf.writeframes(b''.join(audio_frames))
#         wf.close()

#         try:
#             with open("temp.wav", "rb") as audio_file:
#                 transcript = client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file
#                 )
#             user_text = transcript.text
#         except Exception as e:
#             print(f"Transcription error: {e}")
#             self.is_listening = True
#             return

#         print(f"User text: {user_text}")
#         self.text_output.insert(tk.END, f"You: {user_text}\n")
        
#         if self.interrupted_speech:
#             user_text = f"User Interrupted AI Speech: {user_text}"
#             self.interrupted_speech = False

#         self.conversation_context.add_message("user", user_text)

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=self.conversation_context.get_messages()
#         )

#         ai_response = response.choices[0].message.content
#         print(f"AI response: {ai_response}")
#         self.conversation_context.add_message("assistant", ai_response)
#         self.text_output.insert(tk.END, f"AI: {ai_response}\n")

#         self.buffer_queue.clear()

#         tts_response = client.audio.speech.create(
#             model="tts-1",
#             voice="alloy",
#             input=ai_response
#         )

#         audio_segment = AudioSegment.from_mp3(io.BytesIO(tts_response.content))
#         audio_segment = audio_segment.set_frame_rate(24000)
#         audio_array = np.array(audio_segment.get_array_of_samples())
#         stereo_array = np.column_stack((audio_array, audio_array))
#         stereo_array = stereo_array.astype(np.int16)
#         sound = pygame.sndarray.make_sound(stereo_array)
        
#         while any(self.vad.is_speech(frame, 16000) for frame in self.buffer_queue):
#             print("Waiting for user to stop speaking.")
#             time.sleep(0.1)
        
#         self.buffer_queue.clear()

#         with self.tts_lock:
#             self.tts_queue.append({'sound': sound, 'duration': audio_segment.duration_seconds})
#             self.tts_event.set()  # Signal that new TTS is available

#         self.is_listening = True

#     def run(self):
#         self.master.mainloop()

#     def __del__(self):
#         if hasattr(self, 'stream'):
#             self.stream.stop_stream()
#             self.stream.close()
#         self.p.terminate()
#         pygame.quit()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = VADInterruptibleAIAssistant(root)
#     app.run()

import tkinter as tk
import pyaudio
import wave
import threading
import pygame
import io
import os
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
import webrtcvad
import collections
import time

client = OpenAI(api_key="sk-proj-oQpqMVR2jfkrJuWhzhjfT3BlbkFJd41WdvJe6qzW2SXO2dT9")

class ConversationContext:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self):
        return self.messages

class StreamingTTS:
    def __init__(self):
        self.tts_queue = collections.deque()
        self.tts_lock = threading.Lock()
        self.tts_event = threading.Event()
        self.is_speaking = False
        self.current_audio = None
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)
        self.client = OpenAI(api_key="sk-proj-oQpqMVR2jfkrJuWhzhjfT3BlbkFJd41WdvJe6qzW2SXO2dT9")
        # self.initial_buffer_size = 32768  # Adjust this value as needed
        self.initial_buffer_size = 65536  # Adjust this value as needed
        self.buffer_wait_time = 5  # Adjust this value as needed (in seconds)

    def add_to_queue(self, text):
        with self.tts_lock:
            self.tts_queue.append(text)
        self.tts_event.set()

    def tts_player(self):
        while True:
            self.tts_event.wait()
            with self.tts_lock:
                if self.tts_queue:
                    next_tts = self.tts_queue.popleft()
                    self.generate_and_play_streaming(next_tts)
                else:
                    self.tts_event.clear()

    def generate_and_play_streaming(self, text):
        self.is_speaking = True
        
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )

        buffer = io.BytesIO()
        initial_buffer = io.BytesIO()
        is_initial_buffer = True
        start_time = time.time()

        for chunk in response.iter_bytes(chunk_size=4096):
            if not self.is_speaking:
                break

            if is_initial_buffer:
                initial_buffer.write(chunk)
                if initial_buffer.tell() >= self.initial_buffer_size or time.time() - start_time >= self.buffer_wait_time:
                    is_initial_buffer = False
                    buffer = initial_buffer
                    self.play_audio_chunk(buffer)
            else:
                buffer.write(chunk)
                self.play_audio_chunk(buffer)

            buffer.seek(0)
            buffer.truncate(0)

        self.is_speaking = False
        self.current_audio = None

    def play_audio_chunk(self, buffer):
        buffer.seek(0)
        audio = AudioSegment.from_mp3(buffer)
        audio = audio.set_frame_rate(24000)
        
        audio_array = np.array(audio.get_array_of_samples())
        stereo_array = np.column_stack((audio_array, audio_array))
        stereo_array = stereo_array.astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_array)
        sound.play()
        pygame.time.wait(int(audio.duration_seconds * 1000))

    def stop_speaking(self):
        self.is_speaking = False
        pygame.mixer.stop()

    def start(self):
        threading.Thread(target=self.tts_player, daemon=True).start()

class VADInterruptibleAIAssistant:
    def __init__(self, master):
        self.master = master
        master.title("VAD-enabled Interruptible AI Assistant")

        self.conversation_context = ConversationContext()
        self.is_listening = False
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.debounce_time = 1  # 1 second debounce time
        self.interrupted_speech = False
        self.min_audio_length = 0.1  # 100ms minimum audio length for transcription
        self.energy_threshold = 500  # Adjust based on your audio input
        self.silence_threshold = 0.1  # 10% of frames can be silence

        self.text_output = tk.Text(master, height=20, width=50)
        self.text_output.pack()

        self.p = pyaudio.PyAudio()
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)

        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
        self.buffer_queue = collections.deque(maxlen=50)  # 500ms audio buffer
        self.speech_buffer = []

        self.short_interrupt_buffer = collections.deque(maxlen=100)  # 1000ms buffer for short interruptions
        self.tts_paused = False
        self.pause_event = threading.Event()
        
        self.potential_interruption_buffer = []
        self.in_potential_interruption = False
        self.potential_interruption_timer = None

        self.streaming_tts = StreamingTTS()
        self.streaming_tts.start()

        self.start_listening()

    def start_listening(self):
        self.is_listening = True
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=320, stream_callback=self.audio_callback)
        self.stream.start_stream()

    def audio_callback(self, in_data, frame_count, time_info, status):
        frame_duration = 10  # frame duration in ms
        frame_size = int(16000 * frame_duration / 1000 * 2)  # number of bytes per frame
        frames = [in_data[i:i + frame_size] for i in range(0, len(in_data), frame_size)]

        is_speech_detected = False
        for frame in frames:
            if len(frame) == frame_size:
                is_speech = self.vad.is_speech(frame, 16000)
                if is_speech:
                    is_speech_detected = True
                    self.last_speech_time = time.time()
                    self.speech_buffer.append(frame)
                    self.buffer_queue.append(frame)
                    if self.streaming_tts.is_speaking:
                        self.short_interrupt_buffer.append(True)
                        if not self.in_potential_interruption:
                            self.start_potential_interruption()
                        self.potential_interruption_buffer.append(frame)
                        if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and all(self.short_interrupt_buffer):
                            self.confirm_interruption()
                    else:
                        self.short_interrupt_buffer.clear()
                else:
                    self.buffer_queue.append(frame)
                    if self.streaming_tts.is_speaking:
                        self.short_interrupt_buffer.append(False)
                        if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and not any(self.short_interrupt_buffer):
                            self.end_potential_interruption()

        if not is_speech_detected and len(self.buffer_queue) == self.buffer_queue.maxlen:
            current_time = time.time()
            if current_time - self.last_speech_time > self.debounce_time and not self.in_potential_interruption:
                self.process_audio(self.speech_buffer)
                self.speech_buffer.clear()
                self.buffer_queue.clear()

        return (in_data, pyaudio.paContinue)

    def start_potential_interruption(self):
        if not self.in_potential_interruption:
            print("Starting potential interruption.")
            self.in_potential_interruption = True
            self.streaming_tts.stop_speaking()
            self.potential_interruption_timer = threading.Timer(2.0, self.end_potential_interruption)
            self.potential_interruption_timer.start()

    def confirm_interruption(self):
        if self.in_potential_interruption:
            print("Confirming interruption.")
            self.interrupt_speech()
            self.in_potential_interruption = False
            if self.potential_interruption_timer:
                self.potential_interruption_timer.cancel()
            self.potential_interruption_timer = None
            self.process_audio(self.potential_interruption_buffer + self.speech_buffer)
            self.potential_interruption_buffer.clear()
            self.speech_buffer.clear()

    def end_potential_interruption(self):
        if self.in_potential_interruption:
            print("Ending potential interruption.")
            self.in_potential_interruption = False
            if self.potential_interruption_timer:
                self.potential_interruption_timer.cancel()
            self.potential_interruption_timer = None
            self.potential_interruption_buffer.clear()

    def interrupt_speech(self):
        if self.streaming_tts.is_speaking:
            print("Interrupting speech.")
            self.streaming_tts.stop_speaking()
            self.is_speaking = False
            self.interrupted_speech = True
            self.tts_paused = False
            self.text_output.insert(tk.END, "Speech interrupted.\n")

    def is_silence(self, audio_segment):
        return np.sqrt(np.mean(np.square(audio_segment))) < self.energy_threshold

    def contains_speech(self, audio_frames):
        silence_count = sum(1 for frame in audio_frames if self.is_silence(np.frombuffer(frame, dtype=np.int16)))
        return (len(audio_frames) - silence_count) / len(audio_frames) > self.silence_threshold

    def process_audio(self, audio_frames):
        if not audio_frames:
            return

        self.is_listening = False
        print("Processing audio frames.")

        audio_length = len(audio_frames) * 10 / 1000  # in seconds
        if audio_length < self.min_audio_length:
            print(f"Audio too short for transcription: {audio_length} seconds.")
            self.is_listening = True
            return
        
        # Check if audio contains speech
        if not self.contains_speech(audio_frames):
            print("Audio doesn't contain enough speech.")
            self.is_listening = True
            return

        # Write audio frames to WAV file
        wf = wave.open("temp.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

        try:
            with open("temp.wav", "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            user_text = transcript.text
        except Exception as e:
            print(f"Transcription error: {e}")
            self.is_listening = True
            return

        print(f"User text: {user_text}")
        self.text_output.insert(tk.END, f"You: {user_text}\n")
        
        if self.interrupted_speech:
            user_text = f"User Interrupted AI Speech: {user_text}"
            self.interrupted_speech = False

        self.conversation_context.add_message("user", user_text)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_context.get_messages()
        )

        ai_response = response.choices[0].message.content
        print(f"AI response: {ai_response}")
        self.conversation_context.add_message("assistant", ai_response)
        self.text_output.insert(tk.END, f"AI: {ai_response}\n")

        self.streaming_tts.add_to_queue(ai_response)

        self.is_listening = True

    def run(self):
        self.master.mainloop()

    def __del__(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        pygame.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = VADInterruptibleAIAssistant(root)
    app.run()