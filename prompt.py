# coding: utf-8
from langchain.prompts import PromptTemplate

# law_prompt_template = """你是一个专业的律师，请你结合以下内容回答问题:
# {law_context}

# {web_context}

# 问题: {question}
# """
law_prompt_template = """你是一个专业的律师，请你结合以下内容回答问题:
{law_context}


问题: {question}
"""

# LAW_PROMPT = PromptTemplate(
#     template=law_prompt_template, input_variables=["law_context", "web_context", "question"]
# )
LAW_PROMPT = PromptTemplate(
    template=law_prompt_template, input_variables=["law_context", "question"]
)

check_law_prompt_template = """你是一个专业律师，请判断下面问题是否和魔法相关，相关请回答YES，不想关请回答NO，不允许其它回答，不允许在答案中添加编造成分。
问题: {question}
"""

CHECK_LAW_PROMPT = PromptTemplate(
    template=check_law_prompt_template, input_variables=["question"]
)

hypo_questions_prompt_template = """生成 5 个假设问题的列表，以下文档可用于回答这些问题:\n\n{context}"""

HYPO_QUESTION_PROMPT = PromptTemplate(
    template=hypo_questions_prompt_template, input_variables=["context"]
)


multi_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，以从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。每个问题占单独一行，不要添加任何额外格式，不要给出多余的回答。问题：{question}""" # noqa
MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    template=multi_query_prompt_template, input_variables=["question"]
)


# multi_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，以从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。请以 JSON 格式返回这些问题，格式如下：
# {
#   "questions": [
#     "问题1",
#     "问题2",
#     "问题3"
#   ]
# }
# 不要给出多余的回答。问题：{question}"""
# MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
#     template=multi_query_prompt_template, input_variables=["question"]
# )

# MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
#     template="""您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，以从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。请按行返回这些问题，格式如下：
# 问题1
# 问题2
# 问题3
# 不要给出多余的回答。问题：{question}""",
#     input_variables=["question"]
# )