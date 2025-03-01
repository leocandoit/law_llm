

from pprint import pprint
from config import config
from loader import LawLoader
from splitter import MdSplitter
from utils import clear_vectorstore, get_record_manager, law_index


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

if __name__=="__main__":
    init_vectorstore()