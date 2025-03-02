# coding: utf-8
from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field, BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
# from duckduckgo_search.exceptions import DuckDuckGoSearchException

from prompt import MULTI_QUERY_PROMPT_TEMPLATE


# class LawWebRetiever(BaseRetriever):
#     """
#     网页检索其，用于从搜索引擎中检索相关网页，并分割成文档块
#     """
#     # Inputs
#     # 向量存储，用于存储检索到的网页块
#     vectorstore: VectorStore = Field(
#         ..., description="Vector store for storing web pages"
#     )

#     # DuckDuckGo 搜索 API 包装器，用于执行网页搜索
#     search: DuckDuckGoSearchAPIWrapper = Field(..., description="DuckDuckGo Search API Wrapper")
#     # 每次搜索返回的结果数量
#     num_search_results: int = Field(1, description="Number of pages per Google search")

#     # 文本分割器，用于将网页内容分割为块
#     text_splitter: TextSplitter = Field(
#         RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
#         description="Text splitter for splitting web pages into chunks",
#     )

#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun,
#     ) -> List[Document]:
#         """
#         根据查询检索相关文档，并将网页内容分割为文档块。

#         参数:
#             query (str): 用户输入的查询字符串。
#             run_manager (CallbackManagerForRetrieverRun): 回调管理器，用于处理检索过程中的回调。

#         返回:
#             List[Document]: 检索到的文档块列表。
#         """
#         try:
#             # 使用 DuckDuckGo 搜索 API 检索网页
#             results = self.search.results(query, self.num_search_results)
#         except DuckDuckGoSearchException:
#             # 如果搜索失败，返回空列表
#             results = []

#         docs = []
#         for res in results:
#             docs.append(Document(
#                 page_content=res["snippet"],                             # 网页摘要
#                 metadata={"link": res["link"], "title": res["title"]}    # 网页链接和标题
#             ))

#         # 使用文本分割器将文档分割为块
#         docs = self.text_splitter.split_documents(docs)

#         return docs


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


# class LineListOutputParser(PydanticOutputParser):
#     """
#     自定义输出解析器
#     模型的输出解析为按行分隔的列表
#     """
#     def __init__(self) -> None:
#         super().__init__(pydantic_object=LineList)

#     def parse(self, text: str) -> LineList:
#         lines = text.strip().split("\n")
#         return LineList(lines=lines)

class LineListOutputParser(BaseOutputParser):
    """纯文本分行解析器"""
    
    def get_format_instructions(self) -> str:
        return "每行一个结果，不要使用任何格式或标号"
    
    def parse(self, text: str) -> List[str]:
        return [line for line in text.strip().split("\n") if line.strip()]


def get_multi_query_law_retiever(retriever: BaseRetriever, model: BaseModel) -> BaseRetriever:
    """
    多查询检索器
    """
    output_parser = LineListOutputParser() # 创建输出解析器

    llm_chain = LLMChain(llm=model, prompt=MULTI_QUERY_PROMPT_TEMPLATE, output_parser=output_parser)

    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )

    return retriever
