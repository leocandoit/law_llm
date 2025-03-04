
import json
import os
from loader import LawLoader
from splitter import MdSplitter
from utils import get_embedder, get_memory
from langchain_community.vectorstores import Chroma

# 测试分词的效果
def splitterTest():
    LAW_BOOK_CHUNK_SIZE = 233
    LAW_BOOK_CHUNK_OVERLAP = 23
    #新建一个分词器
    md_splitter = MdSplitter.from_tiktoken_encoder(
        chunk_size=LAW_BOOK_CHUNK_SIZE, chunk_overlap=LAW_BOOK_CHUNK_OVERLAP
    )
    # 测试和分割文档
    LAW_BOOK_PATH = r"law_docs"  # 替换为实际路径
    docs = LawLoader(LAW_BOOK_PATH).load_and_split(text_splitter=md_splitter)

    # #结果保存到本地
    # print(f"Total documents: {len(docs)}")  # 打印文档总数
    # for i, doc in enumerate(docs):
    #     print(f"\n=== Document {i+1} ===")
    #     print("Page Content:")
    #     print(doc.page_content)  # 打印文档内容
    #     print("Metadata:")
    #     print(doc.metadata)  # 打印文档元数据
    # #########################下面是保存
    # # 定义保存路径
    # SAVE_PATH = r"test\docs.json"  # 替换为实际路径
    # # 确保目录存在
    # os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # # 将文档转换为字典格式
    # doc_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    # # 保存为 JSON 文件
    # with open(SAVE_PATH, "w", encoding="utf-8") as f:
    #     json.dump(doc_dicts, f, ensure_ascii=False, indent=4)

    # print(f"Documents saved to {SAVE_PATH}")
    return docs


def test_embedding_model():
    # 获取缓存的嵌入模型
    embedder = get_embedder()
    
    # 示例文本
    texts = ["Hello, worl2d!1", "This is a test.1", "Embedding model works.1"]
    
    # 对文本进行嵌入
    embeddings = embedder.encode(texts)
    
    # 打印嵌入结果
    for text, embedding in zip(texts, embeddings):
        print(f"Text: {text}")
        print(f"Embedding: {embedding[:5]}...")  # 只打印前5个维度以节省空间
        print("-" * 40)
    
    # 检查嵌入结果是否非空
    assert all(len(embedding) > 0 for embedding in embeddings), "Embeddings should not be empty."


#打印记忆部分
def test3():
    # 获取记忆
    memory = get_memory()
    print(memory.load_memory_variables({}))
    memory.save_context({"question":"你是谁"},{"answer":"我是LangChain"})
    print(memory.load_memory_variables({}))

    

if __name__ == "__main__":
    test3()