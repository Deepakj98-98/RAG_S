from rag_main import extract_and_chunk
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from typing import TypedDict, Dict, Any
import uuid

REDIS_URI = "redis://localhost:6379"


class DOCState(TypedDict, total=False):
    user_id: str
    thread_id: str
    request: Dict[str, Any]
    response: Dict[str, Any]


async def doc_process_agent(state: DOCState,*,store):
    result = await extract_and_chunk(state["request"])
    await store.aput(
        namespace="docstore",
    key=state["thread_id"],
    value={
        "user_id": state["user_id"],
        "files": state["request"]["file_ids"],
        "status": "processed"
    }
    )
    return {
        "user_id": state["user_id"],
        "thread_id": state["thread_id"],    # FIXED
        "request": state["request"],
        "response": result
    }


def make_config(user_id: str, session_id: str | None = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    thread_id = f"{session_id}"

    return {
        "thread_id": thread_id,
        "user_id": user_id
    }, thread_id


async def docgraph_processor(user_id: str, session_id: str, file_ids: list):
    
    config, thread_id = make_config(user_id, session_id)

    async with (
        AsyncRedisStore.from_conn_string(REDIS_URI) as store,
        AsyncRedisSaver.from_conn_string(REDIS_URI) as checkpointer
    ):
        config["store"] = store
        graph = StateGraph(DOCState)
        graph.add_node("extract", doc_process_agent)

        graph.add_edge(START, "extract")
        graph.add_edge("extract", END)

        agent_app = graph.compile(
            checkpointer=checkpointer,
            store=store
        )

        
        result = await agent_app.ainvoke(
            {
                "user_id": user_id,
                "thread_id": thread_id,
                "request": {"file_ids": file_ids},
                "response": {}
            },
            config
        )
        return result
