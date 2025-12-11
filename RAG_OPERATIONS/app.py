
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from doc_graph import docgraph_processor
import workflow_graph as workflow_graph
from indexing_and_retreival import QdrantOps
from fastapi import FastAPI
import rag_main
from fastapi import Request


app=FastAPI()
indexer=QdrantOps()

@app.on_event("startup")
async def startup_event():
    print("Initialising qdrant")
    await indexer.init_collection()
    print("Qdrant is ready")

@app.post("/dbsave")
async def dbsave(request:Request):
    data=await request.json()
    response=await rag_main.dbsave(data)
    return response

@app.post("/extractandchunk")
async def extract_and_make_chunks(request: Request):
    data =await request.json()
    response=await rag_main.extract_and_chunk(data)
    return response

@app.post("/dbretrieve")
async def db_retrieve(request: Request):
    data =await request.json()
    response=await rag_main.dbretrieve(data)
    return response

@app.post("/retreiveqdrant")
async def qdrantretrieve(request: Request):
    data =await request.json()
    user_query=data.get("user_query")
    if user_query:
        response=await rag_main.retrieveqdrant(user_query=user_query)
        return response
    else:
        "No proper user query received"

@app.post("/videoaudioupload")
async def videoaudioupload(request:Request):
    data=await request.json()
    response=await rag_main.uploads3(data)
    return response

@app.post("/videoaudioprocess")
async def videoaudioprocess(request:Request):
    data=await request.json()
    response=await rag_main.process_videoaudio(data)
    return response

@app.post("/rag_s3_process")
async def rag_s3_process(request:Request):
    data=await request.json()
    user_id=data.get("user_id")
    session_id=data.get("session_id")
    response=await workflow_graph.rag_process(request=data,user_id=user_id,session_id=session_id)
    return response


@app.post("/rag_doc_proccess")
async def rag_doc_proccess(request: Request):
    data=await request.json()
    result=await rag_main.dbsave(data)
    user_id=data.get("user_id")
    session_id=data.get("session_id")
    file_ids=[doc["file_id"] for doc in result["uploaded"]]
    response=await docgraph_processor(user_id=user_id,session_id=session_id,file_ids=file_ids)
    return response


