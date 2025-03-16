
##下面这个可以
from pprint import pprint
from typing import Any, List, Dict
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from config import config
from langchain.indexes._api import _batch
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import Callbacks
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from langchain.memory import ConversationBufferMemory
from openai import OpenAI

import os

os.environ["OPENAI_API_BASE"]="https://chatapi.littlewheat.com/v1"
os.environ["DEEPSEEK_API_BASE"] = "https://api.deepseek.com/v1"
def get_model(callbacks: Callbacks = None):
    return get_model_qwen(Callbacks)

def  get_embeder():
    return get_embedder_bge()

# def get_embedder() -> CacheBackedEmbeddings:
#     fs = LocalFileStore("./.cache/embeddings")
#     underlying_embeddings = OpenAIEmbeddings()
    
#     cached_embedder = CacheBackedEmbeddings.from_bytes_store(
#         underlying_embeddings, fs, namespace=underlying_embeddings.model
#     )
#     return cached_embedder

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


def get_embedder_bge():
    embedder=  HuggingFaceBgeEmbeddings(
        model_name= str(config.EMBEDDING_PATH),
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedder


# 创建并返回一个Chroma向量数据库
def get_vectorstore(collection_name: str = "law") -> Chroma:
    vectorstore = Chroma(
        persist_directory="./chroma_db",        # 持久化存储目录
        embedding_function=get_embeder(),      # 嵌入模型
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


def get_model_openai(
        model: str = "gpt-3.5-turbo-0613",
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatOpenAI:
    model = ChatOpenAI(model=model, streaming=streaming, callbacks=callbacks,temperature=0)
    # temperature=0 禁止创造性回答
    return model


def get_model_qwen(callbacks: Callbacks = None):
    tokenizer = AutoTokenizer.from_pretrained('C:\models\Qwen2.5-3B-Instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('C:\models\Qwen2.5-3B-Instruct', device_map="cuda", trust_remote_code=True).eval()
    pipe1 = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=0.8,
        top_p=0.6,
        repetition_penalty=1.5)
    
    llm = HuggingFacePipeline(pipeline=pipe1,callbacks=callbacks)
    pprint(llm)
    return llm

def get_model_dpsk(
        model: str = "deepseek-chat",
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatOpenAI:
    model = ChatOpenAI(model=model, streaming=streaming, callbacks=callbacks,temperature=0)
    # temperature=0 禁止创造性回答
    return model


# 创建ConversationBufferMemory
def get_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",  # 必须与输入字段名一致
        output_key="answer",    # 必须与输出字段名一致
        return_messages=True
    )


def delete_chroma(collection_name: str = "law",persist_directory: str = "./chroma_db"):
    vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory)
    
    # 删除集合
    vectorstore.delete_collection()
    print(f"Collection '{collection_name}' deleted successfully.")
    

    # 删除集合
    # 因为如果用bge会出现这个问题
    # chromadb.errors.InvalidDimensionException: Embedding dimension 1024 does not match collection dimensionality 1536
    # 维度不匹配，csdn解决方法就是要么删除原来的，要么重新开一个路径





def rerank_documents(question: str, initial_top_n: int = 15, top_n: int = 3) -> List[Dict[str, Any]]:
    # 先使用向量相似搜索找到一些可能相关的文档
    vectorstore = get_vectorstore()
    initial_docs = vectorstore.similarity_search(question, k=initial_top_n)
    # 将这些文档和查询语句组成一个列表，每个元素是一个包含查询和文档内容的列表
    sentence_pairs = [[question, passage.page_content] for passage in initial_docs]

    # 使用FlagReranker模型对这些文档进行重新排序;将use_fp16设置为True可以提高计算速度，但性能略有下降
    reranker = FlagReranker(str(config.RERANKER_PATH))
    # 计算每个文档的得分
    scores = reranker.compute_score(sentence_pairs)

    # 将得分和文档内容组成一个字典列表
    score_document = [{"score": score, "content": content} for score, content in zip(scores, initial_docs)]
    # 根据得分对文档进行排序，并返回前top_n个文档
    result = sorted(score_document, key=lambda x: x['score'], reverse=True)[:top_n]
    print(result)
    return result

def rerank_documents_doc(question: str, initial_top_n: int = 15, top_n: int = 3) -> List[Document]:
    vectorstore = get_vectorstore()
    initial_docs = vectorstore.similarity_search(question, k=initial_top_n)
    sentence_pairs = [[question, passage.page_content] for passage in initial_docs]
    # print("检索内容：")
    # print(sentence_pairs)
    reranker = FlagReranker(str(config.RERANKER_PATH))
    scores = reranker.compute_score(sentence_pairs)

    # 只返回文档，不返回分数
    sorted_docs = [doc for _, doc in sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)[:top_n]]
    # print("排序后：")
    # print(sorted_docs)
    return sorted_docs  # 确保返回的是 List[Document]

if __name__ == "__main__":
    print(rerank_documents(question = "法律"))