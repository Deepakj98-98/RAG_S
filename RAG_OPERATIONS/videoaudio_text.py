import asyncio
import aiofiles
import traceback
from pydub import AudioSegment
from pydub.utils import make_chunks
import whisper
import spacy

class WhisperTranscriptGeneration:
    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp=spacy.load(spacy_model)
        self.whisper_model=whisper.load_model("base")
    
    async def transcribe(self,file_buffer,extension):
        try:
            file_buffer.seek(0)
            audio=await asyncio.to_thread(AudioSegment.from_file,file_buffer,format=extension.lstrip("."))
            chunks=make_chunks(audio,30000)
            async def process_chunk(chunk,idx:int):
                async with aiofiles.tempfile.NamedTemporaryFile(suffix=f"_{idx}.wav",delete=False) as tmpfile:
                    tmp_path=tmpfile.name
                await asyncio.to_thread(chunk.export,tmp_path,format="wav")
                result=await asyncio.to_thread(self.whisper_model.transcribe,tmp_path)
                return result["text"].strip()
            tasks=[process_chunk(chunk,idx) for idx,chunk in enumerate(chunks)]
            results=await asyncio.gather(*tasks)
            return " ".join(results)
        except Exception as e:
            print("Exception occured in transcript generation",traceback.format_exc())
            raise e
        
            



