# coding: utf-8


from pathlib import Path


class Config:
    LAW_BOOK_PATH = "./law_docs"
    LAW_BOOK_CHUNK_SIZE = 100
    LAW_BOOK_CHUNK_OVERLAP = 20
    LAW_VS_COLLECTION_NAME = "law"
    LAW_VS_SEARCH_K = 2

    WEB_VS_COLLECTION_NAME = "web"
    WEB_VS_SEARCH_K = 2

    WEB_HOST = "0.0.0.0"
    WEB_PORT = 7860
    WEB_USERNAME = "username"
    WEB_PASSWORD = "password"
    MAX_HISTORY_LENGTH = 5

    
    ABSOLUTE_PATH = Path(r"C:\models")  # 使用 Path 对象
    EMBEDDING_PATH = ABSOLUTE_PATH / "bge-large-zh-v1.5"  # 使用 / 拼接路径
    RERANKER_PATH = ABSOLUTE_PATH / "bge-reranker-large"

config = Config()
