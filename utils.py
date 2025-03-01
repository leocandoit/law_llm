
##下面这个可以
# from typing import List, Dict
# from collections import defaultdict

# from langchain.docstore.document import Document
# from langchain.storage import LocalFileStore
# from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
# from langchain.indexes import SQLRecordManager, index
# from langchain.vectorstores import Chroma
# from langchain.indexes._api import _batch
# from langchain.chat_models import ChatOpenAI
# from langchain.callbacks.manager import Callbacks


# def get_cached_embedder() -> CacheBackedEmbeddings:
#     fs = LocalFileStore("./.cache/embeddings")
#     underlying_embeddings = OpenAIEmbeddings(openai_api_key="sk-7hBdxF3yd2FEd9r2lvMyX6tJ5X5AYZzqsYFIhwpkTRIr67PF",openai_api_base="https://chatapi.littlewheat.com/v1")
    
#     cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#         underlying_embeddings, fs, namespace=underlying_embeddings.model
#     )
#     return cached_embedder

# if __name__ == "__main__":
#     get_cached_embedder()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings  # 从 langchain.embeddings 导入
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma

#这是通过缓存机制加速处理过程的embedder 最初是用在openai的模型上（上面那个），不知道能不能用在bge上 所以有点问题
def get_cached_embedder1() -> CacheBackedEmbeddings:
    fs = LocalFileStore("./.cache/embeddings")
    model_name = "BAAI/bge-large-zh-v1.5"
    
    model_kwargs = {"device": "cpu"}  # 使用CPU进行计算
    encode_kwargs = {"normalize_embeddings": True}  # 正则化嵌入

    # 使用 HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs,
    )

    # 使用 model_name 作为 namespace
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, fs, namespace=model_name
    )
    # print(cached_embedder)
    return cached_embedder


def get_embedder():
    from FlagEmbedding import FlagAutoModel

    model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                        use_fp16=True)
    return model

# 创建并返回一个Chroma向量数据库
def get_vectorstore(collection_name: str = "law") -> Chroma:
    vectorstore = Chroma(
        persist_directory="./chroma_db",        # 持久化存储目录
        embedding_function=get_embedder(),      # 嵌入模型
        collection_name=collection_name)        # 集合名称 数据库的表

    return vectorstore



if __name__ == "__main__":
    get_vectorstore()