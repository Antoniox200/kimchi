import tkinter as tk
import pyaudio
import wave
import threading
import pygame
import io
import os
import numpy as np
from pydub import AudioSegment
import openai
from openai import OpenAI
import collections
import time
import torch
from scipy import signal
from silero_vad import get_speech_timestamps, load_silero_vad

# Initialize the OpenAI API (replace with your actual API key)
client = OpenAI(api_key="sk-proj-oQpqMVR2jfkrJuWhzhjfT3BlbkFJd41WdvJe6qzW2SXO2dT9")  # Replace with your actual API key

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
        self.stop_event = threading.Event()
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=4096)
        self.initial_buffer_size = 1024 * 64  # 64KB initial buffer
        self.buffer_wait_time = 2  # Wait up to 2 seconds for initial buffer

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
        self.stop_event.clear()
        print("Generating TTS audio...")
        try:
            # Use the OpenAI TTS API
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )

            # Stream the audio content
            audio_chunks = []
            for chunk in response.iter_bytes():
                if self.stop_event.is_set():
                    break
                audio_chunks.append(chunk)

            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                self.play_audio(audio_data)
            self.is_speaking = False
            print("TTS playback finished.")
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            self.is_speaking = False

    def play_audio(self, audio_data):
        buffer = io.BytesIO(audio_data)
        buffer.seek(0)
        audio = AudioSegment.from_file(buffer, format='mp3')
        audio = audio.set_frame_rate(24000).set_channels(2)

        audio_array = np.array(audio.get_array_of_samples())
        stereo_array = audio_array.reshape((-1, 2))
        stereo_array = stereo_array.astype(np.int16)
        sound = pygame.sndarray.make_sound(stereo_array)
        sound.play()
        while pygame.mixer.get_busy():
            if self.stop_event.is_set():
                pygame.mixer.stop()
                break
            pygame.time.wait(100)

    def stop_speaking(self):
        if self.is_speaking:
            print("Stopping TTS playback.")
            self.stop_event.set()
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
        self.debounce_time = 0.1  # Check for speech every 100ms
        self.min_silence_duration = 1.5  # Increased to 1.5 seconds to handle brief pauses
        self.interrupted_speech = False
        self.min_speech_length = .2  # Minimum accumulated speech length for processing

        self.text_output = tk.Text(master, height=20, width=50)
        self.text_output.pack()

        self.p = pyaudio.PyAudio()
        pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)

        # Initialize Silero VAD
        print("Loading Silero VAD model...")
        self.vad_model = load_silero_vad()
        self.vad_samplerate = 16000
        self.window_size_samples = 512  # For 16kHz sampling rate

        self.audio_buffer = bytearray()
        self.buffer_lock = threading.Lock()
        self.processing_audio = False

        self.speech_buffer = bytearray()
        self.potential_interruption_buffer = bytearray()
        self.in_potential_interruption = False

        self.streaming_tts = StreamingTTS()
        self.streaming_tts.start()

        self.start_listening()
        # Start a thread to process audio buffer
        threading.Thread(target=self.process_audio_buffer, daemon=True).start()

    def start_listening(self):
        self.is_listening = True
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.vad_samplerate,
            input=True,
            frames_per_buffer=self.window_size_samples,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print("Started listening...")

    def audio_callback(self, in_data, frame_count, time_info, status):
        # Accumulate audio data
        with self.buffer_lock:
            self.audio_buffer.extend(in_data)
        return (in_data, pyaudio.paContinue)

    def process_audio_buffer(self):
        while True:
            time.sleep(0.1)  # Process every 100ms
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.vad_samplerate * 2 * 0.5:  # At least 0.5 seconds of audio
                    audio_data = self.audio_buffer[:]
                    self.audio_buffer = bytearray()
                else:
                    continue  # Not enough data yet

            # Convert audio_data to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)

            # Apply noise suppression
            audio_np = self.noise_suppression(audio_np)

            # Convert to float32 and normalize
            audio_float32 = audio_np.astype(np.float32) / 32768.0

            # Convert to torch tensor
            torch_audio = torch.from_numpy(audio_float32)

            # Use get_speech_timestamps
            speech_timestamps = get_speech_timestamps(
                torch_audio,
                self.vad_model,
                sampling_rate=self.vad_samplerate,
                threshold=0.5,
                min_speech_duration_ms=100,  # Reduced from 250ms
                min_silence_duration_ms=200,  # Reduced from 500ms
            )

            if speech_timestamps:
                print("Speech detected in buffer.")
                self.last_speech_time = time.time()
                # Extract speech segments and accumulate
                for timestamp in speech_timestamps:
                    start = timestamp['start']
                    end = timestamp['end']
                    start_byte = int(start * 2)  # Multiply by 2 because int16 (2 bytes)
                    end_byte = int(end * 2)
                    speech_segment = audio_data[start_byte:end_byte]
                    self.speech_buffer.extend(speech_segment)
                if self.streaming_tts.is_speaking:
                    if not self.in_potential_interruption:
                        self.start_potential_interruption()
                    self.potential_interruption_buffer.extend(self.speech_buffer)
            else:
                silence_duration = time.time() - self.last_speech_time
                if silence_duration >= self.min_silence_duration and not self.processing_audio:
                    print(f"Silence detected for {silence_duration:.2f} seconds. Processing audio.")
                    self.processing_audio = True
                    threading.Thread(target=self.process_audio, args=(self.speech_buffer.copy(),), daemon=True).start()
                    self.speech_buffer = bytearray()

    def noise_suppression(self, audio_frame):
        # Simple high-pass filter to remove low-frequency noise
        b, a = signal.butter(6, 100 / (self.vad_samplerate / 2), btype='highpass')
        filtered_audio = signal.lfilter(b, a, audio_frame)
        return filtered_audio.astype(np.int16)

    def start_potential_interruption(self):
        if not self.in_potential_interruption:
            print("Starting potential interruption.")
            self.in_potential_interruption = True
            threading.Thread(target=self.confirm_interruption, daemon=True).start()

    def confirm_interruption(self):
        time.sleep(1.0)  # Wait for more potential interruption data
        if self.in_potential_interruption:
            print("Confirming interruption.")
            # Transcribe potential interruption buffer
            transcribed_text = self.transcribe_audio(self.potential_interruption_buffer)
            print(f"Transcribed interruption: {transcribed_text}")

            if self.is_relevant_interruption(transcribed_text):
                self.interrupt_speech()
                self.process_user_input(transcribed_text)
            else:
                print("Interruption not relevant. Continuing TTS playback.")
                self.in_potential_interruption = False
                self.potential_interruption_buffer = bytearray()

            self.in_potential_interruption = False
            self.potential_interruption_buffer = bytearray()

    def is_relevant_interruption(self, transcribed_text):
        # Include recent conversation context
        recent_messages = self.conversation_context.get_messages()[-4:]  # Get the last 4 messages
        context_text = ""
        for msg in recent_messages:
            context_text += f"{msg['role']}: {msg['content']}\n"

        helper_prompt = f"""Consider the following conversation:

    {context_text}

    Now, a potential interruption has occurred.

    The user just said: "{transcribed_text}".

    Based on the conversation so far, is the user trying to interrupt the assistant with something relevant, or is it background noise or irrelevant speech?

    Please respond with only 'interrupt' or 'ignore'. Do not include any explanations or additional text."""

        print("Sending to helper GPT model for assessment.")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": helper_prompt}]
            )
            assessment = response.choices[0].message.content.strip().lower()
            print(f"Helper GPT assessment: {assessment}")
            if assessment == 'interrupt':
                return True
            else:
                return False
        except Exception as e:
            print(f"Helper GPT error: {e}")
            return False


    def interrupt_speech(self):
        if self.streaming_tts.is_speaking:
            print("Interrupting speech.")
            self.streaming_tts.stop_speaking()
            self.is_speaking = False
            self.interrupted_speech = True
            self.text_output.insert(tk.END, "Speech interrupted.\n")

    def process_user_input(self, user_text):
        print(f"User text: {user_text}")
        self.text_output.insert(tk.END, f"You: {user_text}\n")

        if self.interrupted_speech:
            user_text = f"User Interrupted AI Speech: {user_text}"
            self.interrupted_speech = False

        self.conversation_context.add_message("user", user_text)

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_context.get_messages()
            )
            ai_response = response.choices[0].message.content
            print(f"AI response: {ai_response}")
            self.conversation_context.add_message("assistant", ai_response)
            self.text_output.insert(tk.END, f"AI: {ai_response}\n")

            self.streaming_tts.add_to_queue(ai_response)
        except Exception as e:
            print(f"ChatGPT API error: {e}")

    def process_audio(self, audio_data):
        if not audio_data:
            self.processing_audio = False
            return

        print("Processing audio data.")

        audio_length = len(audio_data) / (self.vad_samplerate * 2)  # in seconds
        print(f"Audio length: {audio_length:.2f} seconds")
        if audio_length < self.min_speech_length:
            print(f"Accumulated speech too short for transcription: {audio_length:.2f} seconds.")
            self.processing_audio = False
            return

        # Transcribe audio
        transcribed_text = self.transcribe_audio(audio_data)
        if not transcribed_text:
            print("Transcription failed or empty.")
            self.processing_audio = False
            return

        self.process_user_input(transcribed_text)
        self.processing_audio = False

    def transcribe_audio(self, audio_data):
        # Write audio data to WAV file
        wav_filename = "temp.wav"
        try:
            wf = wave.open(wav_filename, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.vad_samplerate)
            wf.writeframes(audio_data)
            wf.close()

            # Transcribe audio using OpenAI Whisper API
            with open(wav_filename, "rb") as audio_file:
                print("Transcribing audio...")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            transcribed_text = transcript.text
            print(f"Transcription result: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
        finally:
            if os.path.exists(wav_filename):
                os.remove(wav_filename)

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
