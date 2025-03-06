# coding: utf-8
from langchain.prompts import PromptTemplate

# law_prompt_template1 = """你是一个专业的律师，请你结合以下内容回答问题:
# {law_context}

# {web_context}

# 问题: {question}
# """
law_prompt_template = """你是一个专业的律师，请你结合以下内容回答问题:
{law_context}


问题: {question}
"""

# LAW_PROMPT1 = PromptTemplate(
#     template=law_prompt_template, input_variables=["law_context", "web_context", "question"]
# )


LAW_PROMPT = PromptTemplate(
    template=law_prompt_template, input_variables=["law_context", "question"]
)

####添加历史记录#########
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


# CHECK_LAW_PROMPT的核心作用就是将用户输入注入到模板的{question}位置
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


check_intent_prompt_template= """  
你是一个上下文感知的意图分类器，需要结合对话历史分析用户当前问题的完整语义，判断是否属于法律咨询。  

【核心任务】  
1. 分析历史对话中的问题上下文  
2. 补全当前问题的潜在含义（如代词指代、省略内容）  
3. 综合判断最终意图  

【分类规则】  
- 如果当前问题是对历史法律问题的延续或补充 → `law`  
- 如果当前问题独立且不涉及法律 → `other`  
- 如果当前问题需要历史对话才能理解法律含义 → `law`  

【输出要求】  
1. 只输出小写英文标签 `law` 或 `other`  
2. 禁止添加任何解释或标点  

【多轮对话示例】  
用户历史：朋友肇事逃逸怎么办？  
当前问题：其实是我  
→ law（补全含义：其实是我朋友肇事逃逸怎么办）  

用户历史：劳动合同纠纷需要哪些证据？  
当前问题：如果公司不承认怎么办？  
→ law（延续法律场景）  

用户历史：如何追讨拖欠工资？  
当前问题：周末去哪玩？  
→ other（话题切换）  

用户历史：推荐好吃的餐厅  
当前问题：离婚财产怎么分割？  
→ law（独立法律问题）  

【对话历史】  
{chat_history}  

【当前问题】  
{question}  

【综合分析】  
请结合以上历史对话，补全当前问题的完整语义后进行判断：  
"""  
CHECK_INTENT_PROMPT = PromptTemplate(  
    template=check_intent_prompt_template,  
    input_variables=["chat_history", "question"]  
)  





