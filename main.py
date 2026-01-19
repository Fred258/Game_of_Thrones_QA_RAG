# -*- coding: utf-8 -*-
import os

# 导入自定义模块
from config import Config
from modules.data_loader import DocumentProcessor
from modules.vector_store import VectorStoreManager
from modules.rag_engine import RAGEngine
from loguru import logger

# 设置 LangSmith (如果开启)
if Config.ENABLE_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT_NAME
    logger.warning("LangSmith 跟踪已启用,请注意token消耗。")

def format_response(response):
    """美化输出结果"""
    answer = response.get("answer", "无回答")
    sources = response.get("context", [])

    print("\n" + "="*30 + " 🤖 AI 回答 " + "="*30)
    print(answer)
    print("\n" + "="*30 + " 📚 参考文档 " + "="*30)
    if not sources:
        print("未检索到相关文档。")
    else:
        for i, doc in enumerate(sources, 1):
            source_name = os.path.basename(doc.metadata.get('source', '未知'))
            print(f"[{i}] {source_name}")
            print(f"    {doc.page_content.strip()}")
            print(f"Entity: {doc.metadata.get('ner', {})}")
            print("-" * 50)

def main():
    print("=== ⚔️ 权力的游戏 RAG 系统启动中... ===")
    
    # 1. 初始化各管理器
    doc_processor = DocumentProcessor()
    vec_manager = VectorStoreManager()
    
    # 2. 尝试加载现有向量库
    vector_store = vec_manager.load()
    
    # 3. 如果没有向量库，则重新构建
    if not vector_store:
        print("⚠️ 未检测到向量库，开始从文档构建...")
        raw_docs = doc_processor.load_directory(Config.DOCS_DIR)
        if not raw_docs:
            print("❌ 错误：目录下没有可用的文档。")
            return
            
        chunks = doc_processor.split_documents(raw_docs)
        vector_store = vec_manager.save(chunks)
    else:
        print("✅ 成功加载现有向量库。")

    # 4. 初始化 RAG 引擎
    engine = RAGEngine(vector_store)
    qa_chain = engine.build_chain(use_ner_filter=True)

    # 5. 进入交互循环
    print("\n💬 系统就绪！请输入问题 (输入 'exit' 退出)")
    while True:
        query = input("\n用户: ")
        if query.lower() in ['exit', 'quit', '退出']:
            break
            
        try:
            response = qa_chain.invoke({"input": query})
            format_response(response)
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()