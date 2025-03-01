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
后续：还是不行，bge的话缺少方法 AttributeError: 'BaseEmbedder' object has no attribute 'embed_documents'
这表明 self._embedding_function 是一个 BaseEmbedder 对象，但 BaseEmbedder 类没有 embed_documents 方法。这通常是因为嵌入模型（embedding function）未正确设置或初始化。
所以最后还是报错了，等实习的时候问问大佬

2025/3/1
成功往向量化数据库存入数据