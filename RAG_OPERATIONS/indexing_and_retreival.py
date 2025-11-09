import os
import uuid
import numpy as np
import traceback
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams
from sentence_transformers import SentenceTransformer                                                                                                                                                  
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

class QdrantOps:
    def __init__(self):
        self.similarity_threshold=0.5
        self.model=SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.qdrant=AsyncQdrantClient(
            url=os.getenv("ENV_URL"),
            api_key=os.getenv("API_KEY")
        )
        self.collection_name = "document_chunks"
        
    async def init_collection(self):
        try:
            if await self.qdrant.get_collection(self.collection_name):
                print("collection already exists in qdrant")

        except Exception as e:
            try:
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={"size": self.vector_size, "distance": "Cosine"}
                )
                print("Created new collection")
            except Exception as e:
                print("Could not create qdrant collection",traceback.format_exc())
 
    async def semantic_chunking(self,text):
        try:
            sentences=[s.strip() for s in text.split(".") if s.strip()]
            embeddings=self.model.encode(sentences)
            chunks=[]
            current_chunk=[sentences[0]]

            for i in range(1,len(sentences)):
                similarity=np.dot(embeddings[i-1],embeddings[i])/(np.linalg.norm(embeddings[i-1])*np.linalg.norm(embeddings[i]))
                if similarity<self.similarity_threshold:
                    chunks.append(".".join(current_chunk))
                    current_chunk=[sentences[i]]
                else:
                    current_chunk.append(sentences[i])
            chunks.append(".".join(current_chunk))
            return chunks
        except Exception as e:
            print("Exception occurred in semantic chunking",traceback.format_exc())
            raise e

    async def indexing(self,text):
        try:
            points=[]
            try:
                chunks=await self.recursive_chunking(text)
            except Exception as e:
                print("Exception occurred in semantic chunking",traceback.format_exc())
                try:
                    chunks=await self.semantic_chunking(text)
                except Exception as e:
                    print("Exception occurred in recursive chunking",traceback.format_exc())
                    raise e
            embeddings=self.model.encode(chunks,batch_size=32,convert_to_numpy=True)
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"text":chunk}
                )
                for chunk,embedding in zip(chunks,embeddings)
            ]
        except Exception as e:
            print("Exception occurred in indexing",traceback.format_exc())
            raise e

        batch_size=100
        for i in range(0,len(points),batch_size):
            batch=points[i:i+batch_size]
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch)

    async def recursive_chunking(self,text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks=splitter.split_text(text)
        return chunks


    async def retreival_qdrant(self,user_query):
        try:
            vector=self.model.encode(user_query).tolist()
            search_results=await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=None,
                limit=5
            )

            answer=[ans.payload["text"] for ans in search_results.points]
            return "".join(answer)
        except Exception as e:
            print("Exception occurred in qdrant retreival",traceback.format_exc())
            raise e

    async def close(self):
        await self.qdrant.close()


        





