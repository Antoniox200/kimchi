import asyncio
import websockets
import wave
import numpy as np
import requests
import openai

openai.api_key = 'your_openai_api_key'

async def audio_stream(websocket, path):
    async for message in websocket:
        # Process the incoming audio stream
        # Assuming the audio data is received in bytes, convert to numpy array
        audio_data = np.frombuffer(message, dtype=np.int16)
        
        # Save the audio to a file or buffer
        with wave.open('input.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())

        # Use Whisper to transcribe audio to text
        with open('input.wav', 'rb') as audio_file:
            response = openai.Audio.transcribe('whisper-1', audio_file)
            transcription = response['text']
            print(f"Transcription: {transcription}")

        # Use ChatGPT to generate a response
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcription}
            ]
        )
        response_text = chat_response.choices[0].message['content']
        print(f"ChatGPT Response: {response_text}")

        # Convert the response text to speech using OpenAI's TTS API
        tts_response = openai.TextToSpeech.create(
            input=response_text,
            voice="en_us_001",
            output_format="mp3"
        )
        audio_content = tts_response['audio_content']

        # Send the audio response back over the websocket
        await websocket.send(audio_content)

start_server = websockets.serve(audio_stream, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
