import numpy as np
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.indexes import SQLRecordManager, index

from utils import clear_vectorstore, get_embedder_bge, get_record_manager, get_vectorstore

path = "./test_docs/二十届三中全会.docx"
loader = Docx2txtLoader(path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)
docs = text_splitter.split_documents(documents)
chunk_count = len(docs)
# 创建向量数据库
embeddings = get_embedder_bge()

vectorstore = get_vectorstore(collection_name = "bm25")
vectorstore.add_documents(docs)
###########################
# record_manager = get_record_manager("bm25")
# record_manager.create_schema()
# clear_vectorstore("bm25")
# info = index(
#             docs,
#             record_manager,
#             vectorstore,
#             cleanup=None,
#             # cleanup="full",
#             source_id_key="source",
#         )
#############################


# 第一步：创建BM25索引
# 使用文本块创建BM25索引，以实现基于关键词的快速检索。
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

for doc in docs:
    doc.page_content = doc.page_content.replace('\t', ' ') 

bm25 = create_bm25_index(docs) 
# print(bm25)


# 第二步：定义融合检索函数
# 该函数结合了向量搜索和BM25搜索的结果，通过归一化和加权组合得分，返回最相关的文档。

def fusion_retrieval(vectorstore, bm25, query: str, chunk_count: int, k: int = 5, alpha: float = 0.5) -> List[Document]:
    # Step 1: 从vectorstore中获取所有文档
    all_docs = vectorstore.similarity_search("", k=chunk_count)
    # print(all_docs)
    # Step 2: 获取查询的BM25分数
    bm25_scores = bm25.get_scores(query.split())
    print(bm25_scores)
    # Step 3: 使用向量搜索获取相关文档及其分数
    vector_results = vectorstore.similarity_search_with_score(query, k=chunk_count)

    # Step 4: 归一化分数
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # Step 5: 结合分数
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

    # Step 6: 排序文档
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 7: 返回前k个文档
    return [all_docs[i] for i in sorted_indices[:k]]


query = "生态文明"
# print("!!")
top_docs = fusion_retrieval(vectorstore, bm25, query, chunk_count, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]

def show_context(context):
    """
    显示所提供的上下文列表的内容。

    Args:
        context (list):要显示的上下文项列表。

    打印列表中的每个上下文项，并使用指示其位置的标题。
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c)
        print("\n")
# show_context(docs_content)