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

check_law_prompt_template = """你是一名律师，请判断下面问题是否和法律相关，相关请回答YES，不相关请回答NO，不允许其它任何形式的输出，不允许在答案中添加编造成分。
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

formal_question_prompt_template =  """你是一名法律文书助理，请根据以下要求转换问题：
    
    **输入要求**：
    - 原始问题可能包含口语化、模糊或不规范表达
    - 需要符合《中华人民共和国立法法》对法律条文表述的要求
    
    **转换规则**：
    1. 使用完整的主谓宾结构，例如将“打人判几年？”改为“故意伤害他人身体的，应承担何种刑事责任？”
    2. 明确法律主体，如将“公司欠钱不还怎么办？”改为“企业法人未履行债务清偿义务时，债权人可采取哪些法律救济途径？”
    3. 采用法言法语替换口语词汇，例如：
       - "偷东西" -> "盗窃公私财物"
       - "离婚财产" -> "婚姻关系解除后的共同财产分割"
    4. 补充隐含法律要件，如将“酒驾怎么处理？”改为“驾驶机动车时血液酒精含量达到80mg/100ml以上的，应如何依法处置？”
    
    原始问题：{question}
    
    正式法律问题："""
FORMAL_QUESTION_PROMPT = PromptTemplate(
    template=formal_question_prompt_template,input_variables = ["question"]
)