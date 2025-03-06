
import asyncio
import json
from operator import itemgetter
import os
from typing import Any, Dict, List
from callback import OutCallbackHandler
from chain import get_law_chain_intent
from combine import combine_law_docs
from loader import LawLoader
from retriever import get_multi_query_law_retiever
from splitter import MdSplitter
from utils import get_embedder, get_memory, get_model, get_model2, get_vectorstore
from langchain_community.vectorstores import Chroma
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableBranch
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from config import config
from langchain.chains.base import Chain
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

    
async def test_streaming():
    callbacks = [StreamingStdOutCallbackHandler]  # 使用你的 OutCallbackHandler
    prompt = "请简单说明一下法律相关的问题。"
    result = await get_model(callbacks=callbacks)(prompt)
    print("模型返回:", result)

test_prompt_template= """  
你是一个有趣的人，你会在回答问题之后再说一个跟跟问题相关的笑话
【当前问题】  
{question}  

【回答】
"""  
TEST_PROMPT = PromptTemplate(  
    template=test_prompt_template,  
    input_variables=["question"]  
)  


 # model的异步回流测试
async def run_shell() -> None:
    out_callback = OutCallbackHandler1() 
    callbacks = [out_callback] if out_callback else []
    chain = (
        RunnableMap({
            "question": lambda x: x["question"],
            "answer": TEST_PROMPT | get_model2(callbacks=callbacks) | StrOutputParser()
        })
    )
    while True:
        question = input("\n用户:")
        if question.strip() == "stop":
            break

        print("\n法律小助手:", end="")
        # 启动生成任务
        task = asyncio.create_task(chain.ainvoke({"question": question}))
        
        # 并行消费回调输出
        async for new_token in out_callback.aiter():
            print(new_token, end="", flush=True)
        
        res = await task  # 等待生成任务结束
        # 处理res中的其他内容或日志输出
        out_callback.clear()  # 清空队列，重置状态

class OutCallbackHandler1(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[Dict[str, Any]], **kwargs: Any) -> None:
        pass

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queue.put(token)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # 流式输出结束时设置done标识
        self.done.set()
        # 同时可以往队列中放一个终止信号（例如None），或者由aiter()检测done
        await self.queue.put(None)

    async def aiter(self):
        while not self.done.is_set():
            token = await self.queue.get()
            if token is None:
                break
            yield token
        yield "[DONE]"

    def clear(self):
        self.done.clear()
        while not self.queue.empty():
            self.queue.get_nowait()






async def run_shell1() -> None:
    # 1. 初始化链
    # check_law_chain = get_check_law_chain(config)  # 法律问题检查链
    out_callback = OutCallbackHandler1()           # 流式输出回调
    chain = get_law_chain_intent(config, out_callback=out_callback)  # 法律问答链

    # 2. 交互循环
    while True:
        # 3. 接收用户输入
        question = input("\n用户:")
        if question.strip() == "stop":  # 退出条件
            break

        # 4. 检查问题是否与法律相关
        print("\n法律小助手:", end="")
        # is_law = check_law_chain.invoke({"question": question})
        # if not is_law:
            # print("不好意思，我是法律AI助手，请提问和法律有关的问题。")
            # continue  # 跳过非法律问题

        # 5. 生成回答并流式输出
        task = asyncio.create_task(chain.ainvoke({"question": question}))

        async for new_token in out_callback.aiter():  # 逐字输出
            print(new_token, end="", flush=True)

        # 6. 打印完整上下文
        res = await task
        # print(res["law_context"])
        # 7. 重置回调状态
        out_callback.done.clear()

check_intent_prompt_template= """  
你是一个意图分类器，请根据用户问题并结合历史对话的问题进行补全，判断是否属于法律咨询。  

【任务说明】  
1. 只输出分类标签，不要解释  
2. 标签必须为小写英文且无标点  
3. 可选标签：  
   - `law`（法律相关：涉及权利义务、法律法规、诉讼程序等）  
   - `other`（其他：情感、生活技巧、非法律专业问题等）  

【分类示例】  
用户：交通事故责任如何认定？ → law  
用户：怎样追女生？ → other  
用户：劳动合同纠纷怎么解决？ → law  
用户：推荐旅游景点 → other  
【补全示例】
示例一:
用户第一轮：朋友肇事逃逸怎么办？ → law 
用户第二轮: 其实是我 → law  
示例二:
用户第一轮：朋友肇事逃逸怎么办？ → law 
用户第二轮: 如何制定旅行计划 →  other

【历史对话】  
{chat_history}  

【当前问题】  
{question}  

【分类结果】  
"""  
CHECK_INTENT_PROMPT = PromptTemplate(  
    template=check_intent_prompt_template,  
    input_variables=["chat_history", "question"]  
)  
law_prompt_template_history = """  
你是一名专业律师，请严格按以下要求回答问题：  

【回答规则】  
1. 必须基于提供的法律条文，禁止编造  
2. 引用格式：`《法律名称》第XX条`（如：根据《刑法》第264条...）  
3. 若无相关法条，明确告知无法回答  
4. 语言简洁专业，分步骤说明  

【历史对话记录】  
{chat_history}  

【相关法律条文】  
{law_context}  

【用户问题】  
{question}  

【正式回答】  
"""  
LAW_PROMPT_HISTORY = PromptTemplate(  
    template=law_prompt_template_history,  
    input_variables=["chat_history", "law_context", "question"]  
)  



def get_law_chain_intent(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    def is_law_related(x: dict) -> bool:
        # 防御性检查（处理 None 或字段缺失）
        if not x or not isinstance(x, dict) or "intent" not in x:
            print(f" 非法输入: {x}")
            return False
        return str(x.get("intent", "")).strip().lower() == "law"

    # 1. 初始化检索器
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)     # 法律条文向量库
    memory = get_memory()                                       # 内存向量库

    vs_retriever = law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K})  # 法律检索器
    

    # 2. 多查询法律检索器（优化法律条文检索效果）
    multi_query_retriver = get_multi_query_law_retiever(vs_retriever, get_model2())

    # 3. 回调函数配置（用于流式输出）
    callbacks = [out_callback] if out_callback else []
    # 4. 构建链式处理流程
    chain = (
        # 第一步：初始化输入结构
        RunnableMap({
            "question": lambda x: x["question"],
            "chat_history": lambda x: memory.load_memory_variables(x)["chat_history"]
        })
        
        # 第二部 检查意图
        | RunnableMap({
            "intent":CHECK_INTENT_PROMPT | get_model2()|StrOutputParser(),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        })
        | RunnableLambda(lambda x: print(f"[AFTER] 输出数据1: {x}") or x)###############

        
        # 条件分支处理（修复分支数据丢失）
        # 条件分支处理（修复分支数据丢失）
        | RunnableBranch(
            (is_law_related, 
                RunnablePassthrough.assign(  # 保留所有上游变量
                    law_docs = itemgetter("question") | multi_query_retriver
                )
                | RunnableLambda(lambda x: print(f" 检索到法律条文: {len(x['law_docs'])}条") or x)
                | RunnablePassthrough.assign(  # 添加 law_context 并保留其他字段
                    law_context = lambda x: combine_law_docs(x["law_docs"]) or "未找到相关法律"
                )
                | RunnableLambda(lambda x: print(f"[AFTER] 输出数据2: {x}") or x)
                | RunnablePassthrough.assign(  # 动态生成 answer 并保留所有字段
                    answer = RunnablePassthrough.assign(
                        chat_history = itemgetter("chat_history"),
                        law_context = itemgetter("law_context"),
                        question = itemgetter("question")
                    ) | LAW_PROMPT_HISTORY | get_model2(callbacks=callbacks) | StrOutputParser()
                )
            ),
            # 非法律分支（简化处理）
            
             RunnablePassthrough.assign(
                law_context=lambda _: "N/A",
                answer=RunnableLambda(lambda _: "您好，我专注于法律咨询服务。") 
                    | get_model2(callbacks=callbacks)  # 强制流式输出
                    | StrOutputParser())
        )
        
        # 后续处理
        | RunnableLambda(lambda x: (
            memory.save_context({"question": x["question"]}, {"answer": x["answer"]}),
            x
        )[1])
        | RunnableMap({
            "answer": itemgetter("answer"),
            "law_context": itemgetter("law_context")
        })
        
    )

    return chain



if __name__ == "__main__":
    asyncio.run(run_shell1())