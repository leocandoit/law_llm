

import asyncio
from pprint import pprint

import torch
from chain import get_check_law_chain,get_formal_question_chain, get_law_chain
from config import config
from loader import LawLoader
from retriever import LineListOutputParser, get_multi_query_law_retiever, get_multi_query_law_retiever1
from splitter import MdSplitter
from utils import clear_vectorstore, get_model, get_record_manager, get_vectorstore, law_index
from callback import OutCallbackHandler, OutputLogger

#加入向量数据库
def init_vectorstore() -> None:
    record_manager = get_record_manager("law")
    record_manager.create_schema()
    
    clear_vectorstore("law")
    text_splitter = MdSplitter.from_tiktoken_encoder(
        chunk_size=config.LAW_BOOK_CHUNK_SIZE, chunk_overlap=config.LAW_BOOK_CHUNK_OVERLAP
    )
    docs = LawLoader(config.LAW_BOOK_PATH).load_and_split(text_splitter=text_splitter)
    info = law_index(docs)
    pprint(info)  # 以漂亮的方式打印
    """
    打印内容：{'num_added': 145, 'num_deleted': 0, 'num_skipped': 0, 'num_updated': 0}
    成功加入了145个元素
    """


async def run_shell() -> None:
    # 1. 初始化链
    check_law_chain = get_check_law_chain(config)  # 法律问题检查链
    out_callback = OutCallbackHandler()           # 流式输出回调
    chain = get_law_chain(config, out_callback=out_callback)  # 法律问答链

    # 2. 交互循环
    while True:
        # 3. 接收用户输入
        question = input("\n用户:")
        if question.strip() == "stop":  # 退出条件
            break

        # 4. 检查问题是否与法律相关
        print("\n法律小助手:", end="")
        is_law = check_law_chain.invoke({"question": question})
        if not is_law:
            print("不好意思，我是法律AI助手，请提问和法律有关的问题。")
            continue  # 跳过非法律问题

        # 5. 生成回答并流式输出
        task = asyncio.create_task(chain.ainvoke({"question": question}))
        async for new_token in out_callback.aiter():  # 逐字输出
            print(new_token, end="", flush=True)

        # 6. 打印完整上下文
        res = await task
        # print(res["law_context"] + "\n" + res["web_context"])
        print(res["law_context"])
        # 7. 重置回调状态
        out_callback.done.clear()








def testMultiQueryRetriever():
    # 创建多查询检索器
    law_vs = get_vectorstore(config.LAW_VS_COLLECTION_NAME)  # 法律条文向量库
    vs_retriever = law_vs.as_retriever(search_kwargs={"k": config.LAW_VS_SEARCH_K})  # 法律检索器
    multi_query_retriever = get_multi_query_law_retiever1(vs_retriever, get_model())

    # 使用多查询检索器检索文档
    question = "魔法飞行有什么限制吗？"
    documents = multi_query_retriever.get_relevant_documents(question)
    print("成功产生documents")
    # 输出检索结果
    for doc in documents:
        print(doc.page_content)
    
    
def test1():
    parser = LineListOutputParser()
    test_output = "故意伤害罪的构成要件\n伤害案件立案标准\n人身伤害法律量刑"
    print(parser.parse(test_output)) 

async def testGrtLawChain():
    out_callback = OutCallbackHandler()           # 流式输出回调
    chain = get_law_chain(config, out_callback=out_callback)  # 法律问答链
    question = input("\n用户:")
    response = await chain.ainvoke({"question": question})
    print(response)


def test2():
    # 检查非口语的是否可以
    question = "魔法飞行有什么限制吗？"
    chain = get_formal_question_chain(config)
    response = chain.invoke({"question": question});
    pprint(response)





    
if __name__=="__main__":
    # torch.cuda.empty_cache()
    asyncio.run(run_shell())
    # init_vectorstore()
    # testMultiQueryRetriever()
    # test2()