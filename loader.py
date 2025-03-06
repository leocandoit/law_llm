# coding: utf-8
from typing import Any
from langchain_community.document_loaders import TextLoader, DirectoryLoader

class LawLoader(DirectoryLoader):
    """Load law books."""
    def __init__(self, path: str, **kwargs: Any) -> None:
        loader_cls = TextLoader
        glob = "**/*.md"
        loader_kwargs = {"encoding": "utf-8"}
        super().__init__(path, loader_cls=loader_cls, glob=glob,loader_kwargs=loader_kwargs, **kwargs)
