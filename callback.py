# coding: utf-8
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

class OutCallbackHandler(AsyncIteratorCallbackHandler):
    pass

class OutputLogger(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        print("\n=== 模型生成结果 ===")
        print(response.generations[0][0].text)
        print("=====================")