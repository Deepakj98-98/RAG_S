import asyncio
import datetime
import tempfile
import aiofiles
import traceback
from pydub import AudioSegment
from pydub.utils import make_chunks
import whisper
import spacy
import aioboto3
from dotenv import load_dotenv
import os
from io import BytesIO

load_dotenv()

class WhisperTranscriptGeneration:
    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp=spacy.load(spacy_model)
        self.whisper_model=whisper.load_model("base")
        self.bucket_name=os.getenv("AWS_BUCKET_NAME")
        self.aws_region=os.getenv("AWS_REGION")
        self.aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS")
        self.table_name=os.getenv("DYNAMODB_TABLE_NAME")
        print("Whisper Class initiated successfully")
        
    async def transcribe(self,file_buffer,extension):
        try:
            file_buffer.seek(0)
            audio=await asyncio.to_thread(AudioSegment.from_file,file_buffer,format=extension.lstrip("."))
            chunks=make_chunks(audio,30000)
            results=[]
            for idx,chunk in enumerate(chunks):
                async with aiofiles.tempfile.NamedTemporaryFile(suffix=f"_{idx}.wav",delete=False) as tmpfile:
                    tmp_path=tmpfile.name
                await asyncio.to_thread(chunk.export, tmp_path, format="wav")
                result = await asyncio.to_thread(self.whisper_model.transcribe, tmp_path)
                results.append(result["text"].strip())
            print("chunks processed successfully")
            return " ".join(results)
        except Exception as e:
            print("Exception occured in transcript generation",traceback.format_exc())
            raise e

    async def upload_to_s3(self,filepath:str,s3_key:str):
        try:
            session=aioboto3.Session()
            async with session.client("s3",region_name=self.aws_region,aws_access_key_id=self.aws_access_key_id,aws_secret_access_key=self.aws_secret_access_key) as s3:
                await s3.upload_file(filepath,self.bucket_name,s3_key)
                print("File uploaded successfully to s3")
        except Exception as e:
            print("Exception occurred in s3 upload",traceback.format_exc())
            raise e

    async def retrive_from_s3(self,s3_key:str):
        session=aioboto3.Session()
        async with session.client("s3",region_name=self.aws_region,aws_access_key_id=self.aws_access_key_id,aws_secret_access_key=self.aws_secret_access_key) as s3:
            response=await s3.get_object(Bucket=self.bucket_name,Key=s3_key)
            async with response["Body"] as stream:
                data=await stream.read()
                return BytesIO(data)
    
    async def process_s3_file(self,s3_key:str):
        try:
            file_buffer=await self.retrive_from_s3(s3_key)
            extension="."+s3_key.split(".")[-1]
            s3_name=s3_key.split(".")[0]
            s3_transcript_key=f"transcript_{s3_name}.txt"
            transcript=await self.transcribe(file_buffer,extension)
            print("This is the transcript",transcript)
            with tempfile.NamedTemporaryFile(delete=False,suffix=".txt") as tmp:
                tmp.write(transcript.encode("utf-8"))
                tmp_path=tmp.name
            await self.upload_to_s3(tmp_path,s3_key=s3_transcript_key)
            print("transcript complete")
            return {"s3_transcript_key":s3_transcript_key,"transcript":transcript}
        except Exception as e:
            print("Exception occurred in s3 process",traceback.format_exc())
            raise e
        
    async def save_metadata_dynamodb(self,file_id, file_name, s3_key, status="uploaded"):
        session=aioboto3.Session()
        async with session.client(
            "dynamodb",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        ) as client:
            await client.put_item(
                TableName=self.table_name,
                Item={
                    "file_id":{"S":file_id},
                    "file_name":{"S":file_name},
                    "s3_key":{"S":s3_key},
                    "status":{"S":status},
                    "created_at":{"S":datetime.datetime.now(datetime.UTC).isoformat()}
                }
            )
            print("dynamodb save success")

    async def update_dynamodb_metadata(self,file_id,update_data:dict):
        session=aioboto3.Session()
        async with session.client(
            "dynamodb",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        ) as client:
            #table = dynamodb.Table(self.table_name)
            update_expr="SET "+", ".join(f"#{k}=:{k}" for k in update_data.keys())
            print(update_expr)
            expr_attr_names={f"#{k}":k for k in update_data.keys()}
            print(expr_attr_names)
            expr_attr_values={f":{k}":{"S":v} for k,v in update_data.items()}
            print(expr_attr_values)
            await client.update_item(
                TableName=self.table_name,
                Key={"file_id": {"S": file_id}},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_attr_names,
                ExpressionAttributeValues=expr_attr_values
            )
            print("dynamodb update successful")












