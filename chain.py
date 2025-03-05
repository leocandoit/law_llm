
# coding: utf-8
from typing import Any, Optional, List
from collections import defaultdict
from operator import itemgetter
from config import config
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import format_document
from langchain.schema import BaseRetriever
from langchain.pydantic_v1 import Field
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import BooleanOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.chains.base import Chain
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from combine import combine_law_docs, combine_web_docs
from utils import get_memory, get_model2, get_vectorstore, get_model
from retriever import  get_multi_query_law_retiever, get_multi_query_law_retiever1
from prompt import FORMAL_QUESTION_PROMPT, LAW_PROMPT, CHECK_LAW_PROMPT, HYPO_QUESTION_PROMPT, LAW_PROMPT2


def get_check_law_chain(config: Any) -> Chain: 
    model = get_model()                                             # 获取语言模型

    check_chain = CHECK_LAW_PROMPT | model | BooleanOutputParser()  # 构建链式处理流程
    # CHECK_LAW_PROMPT 将用户输入的问题格式化为指定模板的文本 输出：格式化的字符串
    # model 会输入来自上面的字符串 输出相应 根据上面的模板 模型会输出yes 或者 no
    # 这个函数会把yes转换成true no转换成false

    return check_chain 

########下面是测试#####

class DebuggableModel(Runnable):
    def __init__(self, model, debug=True):
        self.model = model
        self.debug = debug

    def invoke(self, input, config=None):
        # 调用原始模型
        output = self.model.invoke(input, config)
        
        # 打印调试信息
        if self.debug:
            print("\n=== 模型原始输出 ===")
            print(f"输入: {input}")
            print(f"输出: {output}")
            print("====================\n")
        
        return output
def get_check_law_chain1(config: Any) -> Chain:
    model = get_model()
    
    # 包装可调试模型
    debuggable_model = DebuggableModel(model, debug=True)  # 调试开关
    
    check_chain = CHECK_LAW_PROMPT | debuggable_model | BooleanOutputParser()
    
    return check_chain


##########上面是测试###########




def get_formal_question_chain(config:Any) -> Chain:
    model = get_model()
    formal_chain = FORMAL_QUESTION_PROMPT | model | StrOutputParser() 
    return formal_chain



def get_law_chain(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    # 1. 初始化检索器
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)  # 法律条文向量库
    # web_vs = get_vectorstore(config.WEB_VS_COLLECTION_NAME)  # 网页内容向量库

    vs_retriever = law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K})  # 法律检索器
    # web_retriever = LawWebRetiever(  # 自定义网页检索器（混合向量+搜索引擎）
    #     vectorstore=web_vs,
    #     search=DuckDuckGoSearchAPIWrapper(),
    #     num_search_results=config.WEB_VS_SEARCH_K
    # )

    # 2. 多查询法律检索器（优化法律条文检索效果）
    multi_query_retriver = get_multi_query_law_retiever(vs_retriever, get_model())

    # 3. 回调函数配置（用于流式输出）
    callbacks = [out_callback] if out_callback else []
    # print("正在检索")
    # 4. 构建链式处理流程
    chain = (
        # 第一步：并行检索法律条文和网页内容
        RunnableMap({
            
            "law_docs": itemgetter("question") | multi_query_retriver,  # 法律条文检索
            # 'web_docs': itemgetter("question") | web_retriever,         # 网页内容检索
            "question": lambda x: x["question"]                        # 保留原始问题
        })
        # 第二步：生成法律和网页的上下文
        | RunnableMap({
            "law_docs": lambda x: x["law_docs"],      # 传递法律文档
            # "web_docs": lambda x: x["web_docs"],      # 传递网页文档
            "law_context": lambda x: combine_law_docs(x["law_docs"]),  # 合并法律文档为上下文
            # "web_context": lambda x: combine_web_docs(x["web_docs"]),  # 合并网页文档为上下文
            "question": lambda x: x["question"]       # 传递问题
        })
        # 第三步：准备提示词模板
        | RunnableMap({
            "law_docs": lambda x: x["law_docs"],      # 传递法律文档
            # "web_docs": lambda x: x["web_docs"],      # 传递网页文档
            "law_context": lambda x: x["law_context"],# 传递法律上下文
            # "web_context": lambda x: x["web_context"],# 传递网页上下文
            "prompt": LAW_PROMPT                     # 插入提示词模板
        })
        # 第四步：调用模型生成回答
        | RunnableMap({
            "law_docs": lambda x: x["law_docs"],      # 传递法律文档
            # "web_docs": lambda x: x["web_docs"],      # 传递网页文档
            "law_context": lambda x: x["law_context"],# 传递法律上下文
            # "web_context": lambda x: x["web_context"],# 传递网页上下文
            "answer": itemgetter("prompt") | get_model(callbacks=callbacks) | StrOutputParser()  # 生成回答
        })
    )
    # print("检索完成")
    return chain

def get_law_chain_history(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    # 1. 初始化检索器
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)  # 法律条文向量库
    memory = get_memory()  # 内存向量库

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
        
        # 第二步：并行检索
        | RunnableMap({
            "law_docs": itemgetter("question") | multi_query_retriver,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        })
        
        # 第三步：构建上下文
        | RunnableMap({
            "law_context": lambda x: combine_law_docs(x["law_docs"]),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        })

        # 第四步：生成回答，同时保留 law_context
        | {
            "answer": LAW_PROMPT2 | get_model2(callbacks=callbacks) | StrOutputParser(),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
            "law_context": itemgetter("law_context")
        }
        
        # 第五步：保存记忆
        | RunnableLambda(lambda x: (memory.save_context(
            {"question": x["question"]},
            {"answer": x["answer"]}
        ), x)[1])
        
        # 第六步：清理输出
        | RunnableMap({
            "answer": itemgetter("answer"),
            "law_context": itemgetter("law_context")
        })
    )

    return chain

def get_hypo_questions_chain(config: Any, callbacks=None) -> Chain:
    # 获取语言模型，支持回调
    model = get_model(callbacks=callbacks)

    # 定义函数调用配置
    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["questions"]
            }
        }
    ]

    # 定义链式操作
    chain = (
        {"context": lambda x: f"《{x.metadata['book']}》{x.page_content}"}
        | HYPO_QUESTION_PROMPT
        | model.bind(functions=functions, function_call={"name": "hypothetical_questions"})
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    return chain
