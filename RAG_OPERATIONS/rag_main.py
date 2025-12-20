
import os
import uuid
from videoaudio_text import WhisperTranscriptGeneration
from pdfdocxtxt_extraction import DocPdfImageExtractor
from dbops import MongoDbTrial
from indexing_and_retreival import QdrantOps
import asyncio
import traceback
from io import BytesIO


async def dbsave(request):
    try:
        mongodbtrail=MongoDbTrial()
        file_paths=request.get("filepaths")
        result=mongodbtrail.save_file(file_paths)
        return {"uploaded":result,"status":"success"}
    except Exception as e:
        print("Exception occurred in dbsave",traceback.format_exc())
        raise e

async def dbretrieve(request):
    try:
        mongodbtrail=MongoDbTrial()
        file_ids=request.get("file_ids")
        files=mongodbtrail.retrieve_by_ids(file_ids)
        return {"status":"success","filenames":list(files.keys())}
    except Exception as e:
        print("Exception occurred in dbretrieve",traceback.format_exc())
        raise e


async def extract_and_chunk(request):
    try:
        mongodbtrail=MongoDbTrial()
        file_ids=request.get("file_ids")
        extractor=DocPdfImageExtractor()
        indexer=QdrantOps()
        file_meta_data,files=mongodbtrail.check_extracted_text(file_ids)
        async def process_file(file:str,file_buffer:BytesIO,file_hash:str):
            try:
                if file.endswith(".docx")or file.endswith(".doc"):
                    text=await extractor.doc_extraction(file_buffer=file_buffer)
                elif file.endswith(".pdf"):
                    text=await extractor.pdf_extraction(file_buffer=file_buffer)
                elif file.endswith(".img") or file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                    text=await extractor.image_text_extraction(file_buffer=file_buffer)
                mongodbtrail.store_extracted_text(
                    {
                        "file_name": file,
                        "file_hash": file_hash,
                        "extracted_text": text,
                        "status": "processed"
                    }
                )
                mongodbtrail.statusfilecheck(
                    {
                        "file_name": file,
                        "status": "success"
                    }
                )
                await indexer.indexing(text)

                return {"file":file,"status":"done"}
            except Exception as e:
                print(f"Error processing {file}: {e}")
                mongodbtrail.store_extracted_text(
                {
                    "file_name": file,
                    "status": "failure"
                })
                return {"file": file, "status": "error", "error": str(e)}
        tasks=[]
        for idx,file_id in enumerate(file_meta_data["file_ids"]):
            file_hash=file_meta_data["file_hash"][idx]
            file,file_buffer=list(files.items())[idx]
            tasks.append(process_file(file,file_buffer=file_buffer,file_hash=file_hash))
        results = await asyncio.gather(*tasks)
        return {
            "status":"success",
            "total_files_processed":len(results),
            "results":results,
            "duplicates":file_meta_data["duplicate_files"]

        }
    except Exception as e:
        print("Exception occurred in extarction/chunking",traceback.format_exc())
        raise e



async def uploads3(request):
    try:
        filepath=request.get("filepath")
        file_id=str(uuid.uuid4())
        filename=os.path.basename(filepath)
        extension=filename.split(".")[-1]
        s3_key=f"{file_id}.{extension}"
        videoaudiotext_extraction=WhisperTranscriptGeneration()
        await videoaudiotext_extraction.upload_to_s3(filepath=filepath,s3_key=s3_key)
        await videoaudiotext_extraction.save_metadata_dynamodb(file_id=file_id,file_name=filename,s3_key=s3_key,status="uploaded")
        return {"file":filename,"s3_key":s3_key,"file_id":file_id}
    except Exception as e:
        print("Exception occurred in s3 upload",traceback.format_exc())
        raise e

async def process_videoaudio(request):
    try:
        
        s3_key=request.get("s3_key")
        file_id=request.get("file_id")
        videoaudiotext_extraction=WhisperTranscriptGeneration()
        transcript_dict=await videoaudiotext_extraction.process_s3_file(s3_key=s3_key)
        update_dict={"s3_transcript_key":transcript_dict["s3_transcript_key"],"status":"processed"}
        await videoaudiotext_extraction.update_dynamodb_metadata(file_id=file_id,update_data=update_dict)
        indexer=QdrantOps()
        await indexer.indexing(transcript_dict["transcript"])
        return transcript_dict
    except Exception as e:
        print("Exception occurred in s3 process",traceback.format_exc())
        raise e








    