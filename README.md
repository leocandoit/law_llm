# law_llm
一个针对法律的大模型项目

# 疑问

```
from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
                                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                      use_fp16=True)
```
query_instruction_for_retrieval和use_fp16在HuggingFaceEmbeddings里没有这个参数设置，FlagAutoModel又用不了，因为我要用CacheBackedEmbeddings，这个为增强性能而设计，通过缓存机制加速处理过程。我现在好像没搞清楚它们之间的关系，所以只使用bge官方的了