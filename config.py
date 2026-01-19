# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class Config:
    
    # === 选定项目名称 === 

    # === 基础路径配置 ===
    # 1. 项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 2. 检索文档根目录
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    # 3. 向量存放目录
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstores")
    
    # 本地向量嵌入模型路径 (请根据实际情况修改)
    EMBEDDING_MODEL_PATH = "D:/RAG/langchain_project/translatorAssistant/basic_knowledge/models/bge-large-zh-v1.5"

    # === LLM 配置 ===
    # API统一使用 OpenAI 接口
    # 1. query改写或后处理模型
    QUERY_REWRITE_MODEL_NAME = "deepseek-chat"
    QUERY_REWRITE_MODEL_API_KEY = os.getenv("QUERY_REWRITE_MODEL_API_KEY")
    QUERY_REWRITE_MODEL_BASE_URL = os.getenv("QUERY_REWRITE_MODEL_BASE_URL")
    QUERY_REWRITE_MODEL_TEMPERATURE = 1.0

    # 2. 回答生成模型
    RESPONSE_MODEL_NAME = "deepseek-chat"
    RESPONSE_MODEL_API_KEY = os.getenv("RESPONSE_MODEL_API_KEY")
    RESPONSE_MODEL_BASE_URL = os.getenv("RESPONSE_MODEL_BASE_URL")
    RESPONSE_MODEL_TEMPERATURE = 0.7

    # 3. 本地模型服务地址
    LOCAL_LLM_SERVICE_PATH = os.getenv("LOCAL_LLM_SERVICE_PATH", None)

    # === 文本切分配置 ===
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 100
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", " ", ""]

    # === 向量库配置 ===
    EMBEDDING_CREATE_BATCH_SIZE = 64


    # === 检索配置 ===
    SEARCH_TYPE = "mmr"
    RETRIEVER_K = 5
    RETRIEVER_FETCH_K = 10
    
    # === LangSmith 配置 (可选) ===
    ENABLE_TRACING = False
    LANGSMITH_PROJECT_NAME = "My First App"

if __name__ == "__main__":
    print("检查所有初始配置项")
    for attr in dir(Config):
        if not attr.startswith("__"):
            print(f"{attr}: {getattr(Config, attr)}")