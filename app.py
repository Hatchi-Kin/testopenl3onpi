from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from utils.extract_openl3_embeddings import EmbeddingsOpenL3
import tensorflow as tf

# Set TensorFlow to only see CPU devices
# Explicitly disable GPU usage as Raspberry Pi does not have a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This line is added to ensure TensorFlow does not attempt to use a GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging, except for errors

# It's recommended to keep run_functions_eagerly for debugging purposes on a Raspberry Pi
# However, for production, you might want to disable this for performance reasons
tf.config.run_functions_eagerly(True)

app = FastAPI(
    title="EmbeddingsOpenL3",
    description="API for extracting music embeddings using OpenL3 and for embeddings similarity with Milvus",
    version="0.1.0",
)

embedding_512_model = EmbeddingsOpenL3("utils/openl3-music-mel128-emb512-3.pb")

class SongList(BaseModel):
    songs: list

class EmbeddingResponse(BaseModel):
    file_name: str
    embedding: list


@app.get("/list_songs", response_model=SongList)
def list_songs():
    source_folder = "music"
    try:
        songs = os.listdir(source_folder)
        return SongList(songs=songs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list songs: {e}")


@app.post("/extract_embeddings_512", response_model=EmbeddingResponse)
def extract_embeddings_512(file_path: str):
    full_path = os.path.join("music", file_path)
    try:
        vector = embedding_512_model.compute(full_path)
        embedding = vector.mean(axis=0)
        return EmbeddingResponse(file_name=file_path, embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting embeddings from {file_path}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)