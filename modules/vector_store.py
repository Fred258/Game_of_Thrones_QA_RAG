# -*- coding: utf-8 -*-
# Vector Store 模块，管理 FAISS 向量库的创建与载入

import os
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from tqdm import tqdm
import json



class VectorStoreManager:
    def __init__(self):
        self.embeddings = None

    def load_embedding_model(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda'}, # 如果没有显卡，会自动回退或需改为 'cpu'
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"已初始化嵌入模型: {Config.EMBEDDING_MODEL_PATH}")


    def save(self, chunks: list, index_name: str = "index", use_embedding_model: bool = True):
        """
        创建新索引并保存至向量库，支持断点续传机制
        Args:
            chunks: 切分后的文档块列表
            index_name: 可自定义索引名称，默认 "index"
        """
        if len(chunks) == 0:
            logger.warning("没有可用于构建向量库的文档块，操作终止。")
            return None

        logger.info(f"为{len(chunks)}个文档块构建向量索引(这可能需要大量时间)...")
        logger.info(f"向量模型为{Config.EMBEDDING_MODEL_PATH.split('/')[-1]}, 每批次处理大小: {Config.EMBEDDING_CREATE_BATCH_SIZE}")
        logger.info(f"这可能需要大量时间，请耐心等待...")

        if use_embedding_model:
            self.load_embedding_model()
        assert self.embeddings is not None, "嵌入模型未正确加载，无法继续构建向量库。"

        # 检测断点
        checkpoint_path = os.path.join(Config.VECTOR_STORE_PATH, f"{index_name}_checkpoint.json")
        start_idx = 0
        vector_store = None

        # --- 1. 检查是否存在断点 ---
        if os.path.exists(checkpoint_path) and os.path.exists(os.path.join(Config.VECTOR_STORE_PATH, f"{index_name}.faiss")):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    start_idx = checkpoint.get("last_processed_idx", 0)
                
                if start_idx < len(chunks):
                    logger.info(f"检测到断点，将从第 {start_idx} 个文档块开始续传...")
                    vector_store = FAISS.load_local(
                        Config.VECTOR_STORE_PATH, 
                        self.embeddings, 
                        index_name=index_name,
                        allow_dangerous_deserialization=True
                    )
            except Exception as e:
                logger.error(f"读取断点文件失败，将重新开始: {e}")
                start_idx = 0


        batch_size = Config.EMBEDDING_CREATE_BATCH_SIZE
        # --- 2. 初始化或继续构建 ---
        with tqdm(total=len(chunks), initial=start_idx, desc="向量化进度") as pbar:
            # 如果是重新开始（start_idx == 0）
            if vector_store is None:
                initial_batch = chunks[:batch_size]
                vector_store = FAISS.from_documents(initial_batch, self.embeddings)
                start_idx = batch_size
                pbar.update(batch_size)
                # 立即保存一次初始状态
                vector_store.save_local(Config.VECTOR_STORE_PATH, index_name=index_name)

            # 循环处理剩余批次
            for i in range(start_idx, len(chunks), batch_size):
                batch = chunks[i : min(i + batch_size, len(chunks))]
                vector_store.add_documents(batch)
                
                # 更新进度
                current_idx = i + len(batch)
                
                # --- 核心：每批次保存 ---
                vector_store.save_local(Config.VECTOR_STORE_PATH, index_name=index_name)
                with open(checkpoint_path, 'w') as f:
                    json.dump({"last_processed_idx": current_idx}, f)
                
                pbar.update(len(batch))

        # --- 3. 完成后清理断点文件 (可选) ---
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("向量库构建全量完成，已清理临时进度文件。")
            
        return vector_store

    def load(self, vector_store_path: str = Config.VECTOR_STORE_PATH, index_name: str = "index", use_embedding_model: bool = True):
        """
        索引已存在于向量库中，直接加载已有索引
        """
        if use_embedding_model:
            if self.embeddings is None:
                self.load_embedding_model()
            assert self.embeddings is not None, "嵌入模型未正确加载，无法继续加载向量库。"
        else:
            self.embeddings = None

        if vector_store_path != Config.VECTOR_STORE_PATH:
            logger.warning(f"传入的向量库路径与向量库默认配置路径不一致!\n请确认是否正确: {vector_store_path} vs {Config.VECTOR_STORE_PATH}")

        try:
            logger.info(f"加载本地向量库: {vector_store_path} -> 索引名称: {index_name} ...")
            
            vector_store = FAISS.load_local(
                folder_path=vector_store_path,
                embeddings=self.embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            logger.info("向量库载入成功")
            return vector_store
        
        except Exception as e:
            logger.error(f"向量库载入异常: {e}")
            return None
        
    # === 新增功能 ===
    def get_documents(self, vector_store):

        """
        直接从内存中的实例获取文档，确保修改能反映到该实例上
        """
        if vector_store is None:
            logger.error("向量库实例为空，无法获取文档。")
            return None
        logger.info(f"共有 {len(vector_store.docstore._dict)} 个文档块存储在向量库中。")
        logger.info(f"document 示例预览: {list(vector_store.docstore._dict.items())[:1]}")
        return vector_store.docstore._dict

    def save_updated_store(self, vector_store, save_path, index_name="index"):
        """保存更新了元数据的向量库"""
        vector_store.save_local(save_path, index_name=index_name)
        logger.info(f"已将更新后的向量库保存至: {save_path}，索引名称: {index_name}")