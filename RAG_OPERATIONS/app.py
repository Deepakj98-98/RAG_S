from indexing_and_retreival import QdrantOps
from fastapi import FastAPI
import rag_main
from fastapi import Request
import uvicorn
import os
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

