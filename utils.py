
##下面这个可以
from pprint import pprint
from typing import List, Dict
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma
from langchain.indexes._api import _batch
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import Callbacks
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline

import os
os.environ["OPENAI_API_KEY"] = "sk-7hBdxF3yd2FEd9r2lvMyX6tJ5X5AYZzqsYFIhwpkTRIr67PF"
os.environ["OPENAI_API_BASE"]="https://chatapi.littlewheat.com/v1"


def get_embedder() -> CacheBackedEmbeddings:
    fs = LocalFileStore("./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings(openai_api_key="sk-7hBdxF3yd2FEd9r2lvMyX6tJ5X5AYZzqsYFIhwpkTRIr67PF",openai_api_base="https://chatapi.littlewheat.com/v1")
    
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder

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


def get_embedder1():
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

#创建一个 SQLRecordManager 实例，用于管理向量数据库中的记录
def get_record_manager(namespace: str = "law") -> SQLRecordManager:
    return SQLRecordManager(
        f"chroma/{namespace}", db_url="sqlite:///law_record_manager_cache.sql"
    )


# 清除向量数据库
def clear_vectorstore(collection_name: str = "law") -> None:
    record_manager = get_record_manager(collection_name)
    vectorstore = get_vectorstore(collection_name)

    index([], record_manager, vectorstore, cleanup="full", source_id_key="source")


# 将 法律相关文档 批量索引 到向量数据库 并进行 记录管理
def law_index(docs: List[Document], show_progress: bool = True) -> Dict:
    info = defaultdict(int)

    record_manager = get_record_manager("law")
    vectorstore = get_vectorstore("law")

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs))

    for docs in _batch(100, docs):
        result = index(
            docs,
            record_manager,
            vectorstore,
            cleanup=None,
            # cleanup="full",
            source_id_key="source",
        )
        for k, v in result.items():
            info[k] += v

        if pbar:
            pbar.update(len(docs))

    if pbar:
        pbar.close()

    return dict(info)


def get_model(
        model: str = "gpt-3.5-turbo-0613",
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatOpenAI:
    model = ChatOpenAI(model=model, streaming=streaming, callbacks=callbacks)
    return model


def get_model1(callbacks: Callbacks = None):
    tokenizer = AutoTokenizer.from_pretrained('C:\models\Qwen2.5-0.5B-Instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('C:\models\Qwen2.5-0.5B-Instruct', device_map="cuda", trust_remote_code=True).eval()
    pipe1 = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.8,
        top_p=0.6,
        repetition_penalty=1.5)
    
    llm = HuggingFacePipeline(pipeline=pipe1,callbacks=callbacks)
    pprint(llm)
    return llm





if __name__ == "__main__":
    get_model()