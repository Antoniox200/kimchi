import asyncio
import websockets
from twilio.twiml.voice_response import VoiceResponse, Start
from fastapi import FastAPI, Request, WebSocket
from pydantic import BaseModel
import openai
import webrtcvad
import wave
import struct

app = FastAPI()

# Initialize OpenAI client (you'll need to set your API key)
openai.api_key = "your_openai_api_key_here"

# Initialize WebRTC VAD
vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)

class AudioChunk(BaseModel):
    chunk: bytes

class ConversationContext:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self):
        return self.messages

async def voice_activity_detector(audio_chunk):
    # Convert audio chunk to 16-bit PCM
    pcm_data = struct.unpack_from("h" * (len(audio_chunk) // 2), audio_chunk)
    
    # Check if there's voice activity
    return vad.is_speech(struct.pack("h" * len(pcm_data), *pcm_data), sample_rate=16000)

async def transcribe_audio(audio_chunk):
    # Save audio chunk to a temporary file
    with wave.open("temp.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_chunk)

    # Use OpenAI's Whisper API for transcription
    with open("temp.wav", "rb") as audio_file:
        transcript = await openai.Audio.atranscribe("whisper-1", audio_file)
    
    return transcript["text"]

async def process_with_chatgpt(transcription, conversation_context):
    conversation_context.add_message("user", transcription)
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=conversation_context.get_messages()
    )
    
    ai_response = response.choices[0].message["content"]
    conversation_context.add_message("assistant", ai_response)
    
    return ai_response

async def text_to_speech(text):
    response = await openai.Audio.acreate(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    return response["audio"]

@app.post("/incoming_call")
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    start = Start()
    start.stream(url=f'wss://{request.host}/stream')
    response.append(start)
    return str(response)

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    conversation_context = ConversationContext()
    tts_task = None
    
    while True:
        audio_chunk = await websocket.receive_bytes()
        
        is_speech = await voice_activity_detector(audio_chunk)
        
        if is_speech:
            if tts_task and not tts_task.done():
                tts_task.cancel()
            
            transcription = await transcribe_audio(audio_chunk)
            response = await process_with_chatgpt(transcription, conversation_context)
            
            async def send_tts():
                speech_audio = await text_to_speech(response)
                await websocket.send_bytes(speech_audio)
            
            tts_task = asyncio.create_task(send_tts())
        else:
            if tts_task and tts_task.done():
                tts_task = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)