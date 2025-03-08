from duckduckgo_search import DDGS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
import requests
from utils import get_model_openai, get_vectorstore

# with DDGS() as ddgs:
#     for r in ddgs.text('床前明月光', region='cn-zh', safesearch='off', timelimit='y', max_results=5):
#         print(r)

def delete_chroma(collection_name: str = "web",persist_directory: str = "./chroma_db"):
    vectorstore = Chroma(collection_name=collection_name, persist_directory=persist_directory)
    
    # 删除集合
    vectorstore.delete_collection()
    print(f"Collection '{collection_name}' deleted successfully.")

delete_chroma()


def search_web(keywords, region='cn-zh', max_results=3):
    web_content = ""
    with DDGS() as ddgs:
        ddgs_gen = ddgs.text(keywords=keywords, region=region, safesearch='off',
                            timelimit='y', max_results=max_results)
        for r in ddgs_gen:
            web_content += (r['body'] + '\n')
        
    return web_content

import requests


import requests
from bs4 import BeautifulSoup


loader = TextLoader('./test_docs/二十届三中全会.txt', 'utf-8')
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document")

# 将文档分割为多个部分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
# docs = text_splitter.split_documents(doc)

# print(docs)
# num_total_characters = sum([len(x.page_content) for x in docs])
# print (f"现在有 {len(docs)} 个文档 平均每个 {num_total_characters / len(docs):,.0f} 字符")
print("*****************************************************")

vectorstore = get_vectorstore("web")
# print(vectorstore.add_documents(docs))

def get_knowledge_based_answer(query,
                               vectorstore,
                               VECTOR_SEARCH_TOP_K,
                               history_len,
                               temperature,
                               top_p,
                               chat_history=[]):
    
    web_content = search_web(query)

    prompt_template = f"""基于以下已知信息，简洁和专业的来回答末尾的问题。
                        如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分。
                        已知网络检索内容：{web_content}""" + """
                        已知本地知识库内容:
                        {context}
                        问题:
                        {question}"""   
        
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    knowledge_chain = RetrievalQA.from_llm(
        llm=get_model_openai(),
        #retriever=vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        retriever=vectorstore.as_retriever(),
        prompt=prompt, 
        verbose=True)    

    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}")

    knowledge_chain.return_source_documents = True


    print(f"-> web_content: {web_content}, prompt: {prompt}, query: {query}" )
    result = knowledge_chain({"query": query})
    return result

query = "怎么学python"
resp = get_knowledge_based_answer(
    query=query,
    vectorstore=vectorstore,
    VECTOR_SEARCH_TOP_K=6,
    chat_history=[],
    history_len=0,
    temperature=0.1,
    top_p=0.9,
)
print(resp)

