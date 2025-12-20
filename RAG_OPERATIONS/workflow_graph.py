import asyncio
import uuid
from pydantic import FilePath
from rag_main import process_videoaudio, uploads3
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
#from langgraph.store.postgres import PostgresStore
#from langgraph.checkpoint.postgres import PostgresSaver 
from langgraph.store.base import BaseStore
from typing import TypedDict, List, Dict, Optional
#from langgraph.store.postgres.aio import AsyncPostgresStore
#from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
#from langgraph.checkpoint.redis import RedisSaver 
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from indexing_and_retreival import QdrantOps
import aiohttp

REDIS_URI = "redis://localhost:6379"

class RAGState(TypedDict,total=False):
    user_id:str
    thread_id:str
    history:List[Dict[str,str]]
    filepath: Optional[str]
    file_id: Optional[str]
    s3_key: Optional[str]
    s3_transcript_key:Optional[str]
    transcript:Optional[str]
    question:Optional[str]
    answer:Optional[str]

#DB_URI = "postgresql://postgres:deepak123@localhost:5432/langgraph_db?sslmode=disable"

async def upload_agent(state: RAGState,*,store):
    request={"filepath":state["filepath"]}
    result=await uploads3(request=request)
    user_id=state["user_id"]
    await store.aput(
        ("uploads",user_id,result["file_id"]),
        "upload",
        {
        "file_id": result["file_id"],
        "s3_key": result["s3_key"],
        "filename": result.get("file")
        }
    )

    return {
        "file_id": result["file_id"],
        "s3_key": result["s3_key"],
        "filename":result.get("file"),
        "history": state.get("history", []) + [
            {"role": "system", "content": "File uploaded successfully"}
        ]
    }

async def process_agent(state:RAGState,*,store):
    request = {
        "file_id": state["file_id"],
        "s3_key": state["s3_key"]
    }
    result=await process_videoaudio(request)
    user_id=state["user_id"]
    await store.aput(
        ("transcripts",user_id,state["file_id"]),
        "latest",
        {
        "data":result["transcript"]
        }
    )
    return {
        "s3_transcript_key": result["s3_transcript_key"],
        "transcript": result["transcript"],
        "history": state.get("history",[]) + [
            {"role": "system", "content": "Transcript generated and indexed"}
        ]
    }


def make_config(user_id: str, session_id: str | None = None):
    if session_id is None:
        session_id = str(uuid.uuid4())

    return {
        "configurable": {
            "user_id": user_id,
            "thread_id": f"{user_id}-{session_id}"
        }
    }

async def query_ollama(model, prompt):
    url="http://localhost:11434/api/chat"
    payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url,json=payload) as response:
                data=await response.json()
                return data["message"]["content"].strip()
      

async def retrieveqdrant(user_query):
    qdrant_retrieve=QdrantOps()
    response=await qdrant_retrieve.retreival_qdrant(user_query=user_query)
    return response


async def answer_agent(state: RAGState, *, store):

    docs = state["retrieved_docs"]
    question = state["question"]

    prompt = f"""
    Answer the question using the context below.
    Context:
    {docs}

    Question:
    {question}
    """

    response = await query_ollama("mistral",prompt=prompt)

    return {
        "answer": response.content,
        "history": state.get("history", []) + [
            {"role": "assistant", "content": response.content}
        ]
    }


async def retrieve_agent(state: RAGState, *, store):
    user_query = state["question"]
    user_id = state["user_id"]

    # Call Qdrant
    response = await retrieveqdrant(user_query)

    # Persist retrieved results
    await store.aput(
        ("qdrant_retrieval", user_id),
        "latest",
        {
            "query": user_query,
            "results": response
        }
    )

    return {
        "retrieved_docs": response,
        "history": state.get("history", []) + [
            {"role": "system", "content": "Documents retrieved from Qdrant"}
        ]
    }



async def rag_process(request,user_id,session_id:Optional[str]=None):
    filepath=request.get("filepath")
    async with (
        AsyncRedisStore.from_conn_string(REDIS_URI) as store,  
        AsyncRedisSaver.from_conn_string(REDIS_URI) as checkpointer,
    ):
        #await store.setup()
        #await checkpointer.asetup()
        rag_processor=StateGraph(RAGState)
        rag_processor.add_node("upload",upload_agent)
        rag_processor.add_node("process",process_agent)
        rag_processor.add_edge(START,"upload")
        rag_processor.add_edge("upload","process")
        rag_processor.add_edge("process",END)
        graph=rag_processor.compile(
            checkpointer=checkpointer,
            store=store
        )

        config = make_config(user_id=user_id,session_id=session_id)
        result =await graph.ainvoke(
        {
            "filepath": filepath,
            "user_id":user_id,
            "history": []
        },
        config
    )
        return result
        
async def qa_process(request, user_id, session_id: Optional[str] = None):
    question = request.get("question")
    async with (
        AsyncRedisStore.from_conn_string(REDIS_URI) as store,
        AsyncRedisSaver.from_conn_string(REDIS_URI) as checkpointer,
    ):
        qa_graph=StateGraph(RAGState)
        qa_graph.add_node("retrieve",retrieve_agent)
        qa_graph.add_node("answer",answer_agent)
        qa_graph.add_edge(START, "retrieve")
        qa_graph.add_edge("retrieve", "answer")
        qa_graph.add_edge("answer", END)

        graph = qa_graph.compile(
            checkpointer=checkpointer,
            store=store
        )

        config = make_config(user_id=user_id, session_id=session_id)

        result = await graph.ainvoke(
            {
                "question": question,
                "user_id": user_id,
                "history": []
            },
            config
        )

        return result

    
# event_loop_thread.py  (or at top of workflow_graph.py)
import asyncio
from threading import Thread

# create and start a background loop once
_BG_LOOP = None

def start_background_loop():
    global _BG_LOOP
    if _BG_LOOP is not None:
        return _BG_LOOP
    loop = asyncio.new_event_loop()
    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    t = Thread(target=run, daemon=True)
    t.start()
    _BG_LOOP = loop
    return _BG_LOOP

def submit_coro(coro):
    """
    Submit coroutine to the background loop and block until result.
    Raises exceptions from the coroutine.
    """
    loop = start_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
 