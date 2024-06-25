# Updated Program with New Prompts for CHATGPT:

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
#         self.debounce_time = 0.5  # 500ms debounce time
#         self.interrupted_speech = False  # New flag to track if speech was interrupted

#         self.text_output = tk.Text(master, height=20, width=50)
#         self.text_output.pack()

#         self.p = pyaudio.PyAudio()
#         pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)

#         self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
#         self.buffer_queue = collections.deque(maxlen=50)  # 500ms audio buffer
#         self.speech_buffer = []

#         self.start_listening()

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
#                 print(f"Audio Callback - Is speech: {is_speech}")
#                 if is_speech:
#                     is_speech_detected = True
#                     self.last_speech_time = time.time()
#                     self.speech_buffer.append(frame)
#                     self.buffer_queue.append(frame)
#                     if self.is_speaking:
#                         self.interrupt_speech()
#                 else:
#                     self.buffer_queue.append(frame)

#         if not is_speech_detected and len(self.buffer_queue) == self.buffer_queue.maxlen:
#             current_time = time.time()
#             if current_time - self.last_speech_time > self.debounce_time:
#                 print("Processing audio due to buffer queue max length.")
#                 self.process_audio(self.speech_buffer)
#                 self.speech_buffer.clear()
#                 self.buffer_queue.clear()

#         return (in_data, pyaudio.paContinue)

#     def interrupt_speech(self):
#         if self.is_speaking:
#             print("Interrupting speech.")
#             pygame.mixer.stop()
#             self.is_speaking = False
#             self.interrupted_speech = True  # Set the flag when speech is interrupted
#             self.text_output.insert(tk.END, "Speech interrupted.\n")

#     def play_audio(self, sound, duration):
#         self.is_speaking = True
#         sound.play()
#         self.master.after(int(duration * 1000), self.stop_speaking)

#     def stop_speaking(self):
#         pygame.mixer.stop()
#         self.is_speaking = False

#     def process_audio(self, audio_frames):
#         if not audio_frames:
#             return

#         self.is_listening = False
#         print("Processing audio frames.")

#         # Save recorded audio to a WAV file
#         wf = wave.open("temp.wav", "wb")
#         wf.setnchannels(1)
#         wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(16000)
#         wf.writeframes(b''.join(audio_frames))
#         wf.close()

#         # Ensure the audio length is at least 0.1 seconds before transcribing
#         audio_length = len(audio_frames) * 10 / 1000  # in seconds
#         if audio_length < 0.1:
#             print("Audio too short for transcription.")
#             self.is_listening = True
#             return

#         try:
#             # Transcribe audio
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
        
#         # Check if this was an interruption and modify the message accordingly
#         if self.interrupted_speech:
#             user_text = f"User Interrupted AI Speech: {user_text}"
#             self.interrupted_speech = False  # Reset the flag
        
#         self.conversation_context.add_message("user", user_text)

#         # Process with ChatGPT
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=self.conversation_context.get_messages()
#         )

#         ai_response = response.choices[0].message.content
#         print(f"AI response: {ai_response}")
#         self.conversation_context.add_message("assistant", ai_response)
#         self.text_output.insert(tk.END, f"AI: {ai_response}\n")

#         self.buffer_queue.clear()  # Clear the buffer queue to reset VAD state

#         # Convert AI response to speech
#         tts_response = client.audio.speech.create(
#             model="tts-1",
#             voice="alloy",
#             input=ai_response
#         )

#         # Play the audio response
#         audio_segment = AudioSegment.from_mp3(io.BytesIO(tts_response.content))
#         audio_segment = audio_segment.set_frame_rate(24000)
#         audio_array = np.array(audio_segment.get_array_of_samples())
#         stereo_array = np.column_stack((audio_array, audio_array))
#         stereo_array = stereo_array.astype(np.int16)
#         sound = pygame.sndarray.make_sound(stereo_array)
        
#         # Ensure no user speech is detected before playing AI response
#         while any(self.vad.is_speech(frame, 16000) for frame in self.buffer_queue):
#             print("Waiting for user to stop speaking.")
#             time.sleep(0.1)  # Wait for 100ms before checking again
        
#         self.buffer_queue.clear()  # Clear the buffer queue to reset VAD state

#         threading.Thread(target=self.play_audio, args=(sound, audio_segment.duration_seconds)).start()

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

class VADInterruptibleAIAssistant:
    def __init__(self, master):
        self.master = master
        master.title("VAD-enabled Interruptible AI Assistant")

        self.conversation_context = ConversationContext()
        self.is_listening = False
        self.is_speaking = False
        self.last_speech_time = time.time()
        self.debounce_time = 0.5  # 500ms debounce time
        self.interrupted_speech = False

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
                print(f"Audio Callback - Is speech: {is_speech}")
                if is_speech:
                    is_speech_detected = True
                    self.last_speech_time = time.time()
                    self.speech_buffer.append(frame)
                    self.buffer_queue.append(frame)
                    if self.is_speaking:
                        self.short_interrupt_buffer.append(True)
                        if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and all(self.short_interrupt_buffer):
                            self.interrupt_speech()
                        else:
                            print("Short interruption detected.")
                            self.pause_tts()
                    else:
                        self.short_interrupt_buffer.clear()
                else:
                    self.buffer_queue.append(frame)
                    if self.is_speaking:
                        self.short_interrupt_buffer.append(False)
                        if len(self.short_interrupt_buffer) == self.short_interrupt_buffer.maxlen and not any(self.short_interrupt_buffer):
                            self.resume_tts()

        if not is_speech_detected and len(self.buffer_queue) == self.buffer_queue.maxlen:
            current_time = time.time()
            if current_time - self.last_speech_time > self.debounce_time:
                print("Processing audio due to buffer queue max length.")
                self.process_audio(self.speech_buffer)
                self.speech_buffer.clear()
                self.buffer_queue.clear()

        return (in_data, pyaudio.paContinue)

    def interrupt_speech(self):
        if self.is_speaking:
            print("Interrupting speech.")
            pygame.mixer.stop()
            self.is_speaking = False
            self.interrupted_speech = True
            self.tts_paused = False
            self.text_output.insert(tk.END, "Speech interrupted.\n")

    def pause_tts(self):
        if self.is_speaking and not self.tts_paused:
            print("Pausing speech.")
            pygame.mixer.pause()
            self.tts_paused = True
            self.pause_event.set()

    def resume_tts(self):
        if self.is_speaking and self.tts_paused:
            print("Resuming speech.")
            pygame.mixer.unpause()
            self.tts_paused = False
            self.pause_event.clear()

    def play_audio(self, sound, duration):
        self.is_speaking = True
        sound.play()
        start_time = time.time()
        elapsed_time = 0
        while elapsed_time < duration:
            if self.pause_event.is_set():
                pygame.mixer.pause()
                self.pause_event.wait()  # Wait until the pause_event is cleared
                pygame.mixer.unpause()
                start_time = time.time() - elapsed_time  # Adjust start_time
            elapsed_time = time.time() - start_time
            time.sleep(0.1)  # Sleep to prevent busy-waiting
        self.stop_speaking()

    def stop_speaking(self):
        pygame.mixer.stop()
        self.is_speaking = False

    def process_audio(self, audio_frames):
        if not audio_frames:
            return

        self.is_listening = False
        print("Processing audio frames.")

        # Save recorded audio to a WAV file
        wf = wave.open("temp.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

        # Ensure the audio length is at least 0.1 seconds before transcribing
        audio_length = len(audio_frames) * 10 / 1000  # in seconds
        if audio_length < 0.1:
            print("Audio too short for transcription.")
            self.is_listening = True
            return

        try:
            # Transcribe audio
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
        
        # Check if this was an interruption and modify the message accordingly
        if self.interrupted_speech:
            user_text = f"User Interrupted AI Speech: {user_text}"
            self.interrupted_speech = False  # Reset the flag
        
        self.conversation_context.add_message("user", user_text)

        # Process with ChatGPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.conversation_context.get_messages()
        )

        ai_response = response.choices[0].message.content
        print(f"AI response: {ai_response}")
        self.conversation_context.add_message("assistant", ai_response)
        self.text_output.insert(tk.END, f"AI: {ai_response}\n")

        self.buffer_queue.clear()  # Clear the buffer queue to reset VAD state

        # Convert AI response to speech
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ai_response
        )

        # Play the audio response
        audio_segment = AudioSegment.from_mp3(io.BytesIO(tts_response.content))
        audio_segment = audio_segment.set_frame_rate(24000)
        audio_array = np.array(audio_segment.get_array_of_samples())
        stereo_array = np.column_stack((audio_array, audio_array))
        stereo_array = stereo_array.astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_array)
        
        # Ensure no user speech is detected before playing AI response
        while any(self.vad.is_speech(frame, 16000) for frame in self.buffer_queue):
            print("Waiting for user to stop speaking.")
            time.sleep(0.1)  # Wait for 100ms before checking again
        
        self.buffer_queue.clear()  # Clear the buffer queue to reset VAD state

        threading.Thread(target=self.play_audio, args=(sound, audio_segment.duration_seconds)).start()

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