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

# 多查询检索
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


check_intent_prompt_template = """  
# 法律意图识别分类器

## 核心任务
分析对话上下文，判断当前问题是否属于法律咨询场景

## 分类标准
`law` 需同时满足：
1. 当前问题包含法律要素（权利义务/纠纷解决/法律程序）
2. 满足以下任意条件：
   - 与历史中的法律咨询构成连续对话
   - 需要法律背景才能准确理解
   - 明确涉及法律实体（如合同/诉讼/婚姻）

`other` 需满足：
1. 问题不涉及法律要素
2. 或属于日常闲聊/事实查询/其他领域咨询

## 处理流程
1. 扫描最近3轮对话，标记法律相关实体
2. 检测当前问题的法律关键词
3. 判断上下文关联性
4. 最终分类决策

## 输出规范
仅返回小写标签，禁止任何解释
→ 合法咨询：`law`
→ 其他场景：`other`

## 典型场景示例
├─ 延续法律咨询
历史：交通事故赔偿标准
当前：伤残鉴定怎么做？
→ law

├─ 隐含法律场景  
历史：空白
当前：微信被辞退有补偿吗？
→ law

├─ 法律术语触发
历史：讨论周末聚会
当前：协议离婚需要冷静期吗？
→ law

├─ 非法律咨询  
历史：合同纠纷准备材料  
当前：推荐律所周边餐厅  
→ other

## 输入数据
[对话历史]（最多3轮）
{chat_history}

[当前问题]
{question}

## 识别判断
"""  
CHECK_INTENT_PROMPT = PromptTemplate(  
    template=check_intent_prompt_template,  
    input_variables=["chat_history", "question"]  
)  


FRIENDLY_REJECTION_PROMPT_template = """
[用户问题]
{question}

[对话延续规则]  
1️⃣ **情感共鸣**：先回应原始问题的情感价值  
2️⃣ **场景关联**：挖掘该话题下的潜在法律需求
3️⃣ **开放引导**：给用户提供咨询方向

[示例指令]  
用户：最近想辞职去旅游  
→ "放松身心确实很重要呢！(😊) 在规划旅程时，是否需要了解：  
• 离职期间的劳动权益保障  
• 旅游合同中的消费者保护条款  
• 旅途意外伤害的法律责任划分"  


用户：推荐周末活动  
→ "休闲活动能缓解压力呢~(🌿) 如果涉及：  
• 活动场地的安全责任  
• 预付卡消费纠纷  
• 人身意外保险索赔  
这些法律知识可能会帮到您！
"""
FRIENDLY_REJECTION_PROMPT=PromptTemplate(  
    template=FRIENDLY_REJECTION_PROMPT_template,  
    input_variables=["question"]  
)  

pre_question_prompt_template = """  
你是一名对话上下文分析师，请完善当前输入内容：

# 处理标准
▧ 需要补全（满足任一）：
① 含模糊指代（这/那/上述/他/她/它）
② 依赖对话中的特定对象/事件
③ 缺少必要情境要素
④ 延续之前的对话线程

▧ 保持原样（满足任一）：
✓ 信息完整自洽
✓ 独立操作指令
✓ 明确的新话题声明

# 输入类型处理规范
[对话记录]（最近3条）  
{chat_history}  

[当前输入]  
{question}  

# 多场景处理示例
■ 疑问句补全：  
输入："这个方案可行吗？"  
上下文：讨论过市场推广方案  
→ "昨天提出的市场推广方案在预算范围内可行吗？"  

■ 陈述句补全：  
输入："数据需要重新校验"  
上下文：前文提及Q3财报数据  
→ "Q3财报的销售数据需要重新校验"  

■ 祈使句补全：  
输入："发给我最终版"  
上下文：正在修订合作协议  
→ "请将合作协议的最终版发给我"  

■ 保持原样案例：  
输入："创建新的会议日程"  
上下文：无相关讨论  
→ "创建新的会议日程"  

# 执行流程
1. 检测是否存在语境依赖
2. 需要补全时：
   - 锚定上下文中的关联要素
   - 保持原始语义和语气
   - 输出「完整版：」开头的优化内容
3. 无需补全时：
   - 输出「原始版：」开头的原文

当前任务处理：
------------------
[对话记录]  
{chat_history}  

[当前输入]  
{question}  

处理结果（直接输出，无需标记）：
"""  

PRE_QUESTION_PROMPT = PromptTemplate(  
    template=pre_question_prompt_template,  
    input_variables=["chat_history", "question"]
)  