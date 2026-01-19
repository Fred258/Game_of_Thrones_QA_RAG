# -*- coding: utf-8 -*-
import json
import os
import time
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import Config
from modules.vector_store import VectorStoreManager
from modules.prompts import RAGPrompts
from loguru import logger

class NERProcessor:
    def __init__(self, vector_store_path, index_name="index"):
        self.index_name = index_name
        self.vector_store_path = vector_store_path
        self.vec_manager = VectorStoreManager()
        
        logger.info("🔄 加载向量库,NER任务无需采用embedding模式...")
        # 1. 加载向量库 (此时它是硬盘上的旧状态)
        self.vector_store = self.vec_manager.load(
            vector_store_path=self.vector_store_path,
            index_name=self.index_name,
            use_embedding_model=False
        )
        
        # 2. 定义文件路径
        self.checkpoint_file = os.path.join(Config.DOCS_DIR, "ner_checkpoint.json")
        self.cache_file = os.path.join(Config.DOCS_DIR, "ner_temp_cache.json")

        # 3. 初始化状态 (核心逻辑)
        # committed_ids: 已经确保存入 FAISS 文件的 ID
        # cached_data:   已经识别完但只存在 json 里的临时数据 {doc_id: ner_result}
        self.committed_ids, self.cached_data = self._load_state()

        # 4. 将缓存中的数据“回放”到内存中的向量库
        # 这一步至关重要：虽然 FAISS 文件没存，但我们把上次崩溃前缓存的 NER 结果重新注入内存
        if self.cached_data:
            logger.info(f"🔄 正在回放 {len(self.cached_data)} 条缓存数据到内存向量库...")
            self._apply_cache_to_memory(self.cached_data)

        # 5. 初始化模型 (保持不变)
        if not Config.LOCAL_LLM_SERVICE_PATH:
            raise ValueError("请配置Ollama LOCAL_LLM_SERVICE_PATH")
            
        self.local_llm = ChatOllama(
            model="qwen2.5:1.5b-instruct", 
            temperature=0.1, # NER 任务建议低温
            format="json", 
            base_url="http://127.0.0.1:11434",
            timeout=60,
            num_predict=512
        )
        self.parser = JsonOutputParser()
        self.prompt = PromptTemplate(template=RAGPrompts.NER_TEMPLATE, input_variables=["text"])
        self.chain = self.prompt | self.local_llm | self.parser

    def _load_state(self):
        """
        启动时读取状态
        Returns:
            committed_ids (set): 向量库中已固化的
            cached_data (dict): 临时文件中的 {id: result}
        """
        committed = set()
        cached = {}

        # 读取 Checkpoint (记录 ID 分组)
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    committed = set(data.get("committed_ids", []))
                    # cached_ids = set(data.get("cached_ids", [])) # 其实不需要读这个 list，直接读 cache 文件更准
            except Exception as e:
                logger.error(f"读取 Checkpoint 失败: {e}")

        # 读取 Cache Data (记录实际 NER 内容)
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
            except Exception as e:
                logger.error(f"读取缓存数据失败: {e}")
        
        return committed, cached

    def _apply_cache_to_memory(self, cache_dict):
        """将缓存数据注入当前内存中的 VectorStore"""
        docstore = self.vec_manager.get_documents(self.vector_store)
        if not docstore:
            logger.error("向量库实例为None，无法实现数据ner更新。")
            return
        
        count = 0
        for doc_id, ner_result in cache_dict.items():
            if doc_id in docstore:
                docstore[doc_id].metadata["ner"] = ner_result
                count += 1
        logger.info(f"✅ 已恢复 {count} 条缓存记录到内存。")

    def _save_temp_state(self):
        """
        【小步频保存】
        只保存 checkpoint 和 cache.json，不碰 FAISS
        速度快，开销小
        """
        try:
            # 1. 保存 NER 结果内容
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cached_data, f, ensure_ascii=False)
            
            # 2. 保存 ID 状态
            state = {
                "committed_ids": list(self.committed_ids),
                "cached_ids": list(self.cached_data.keys()) # 这些是已处理但未入库的
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(state, f)
                
            # logger.debug(f"⚡ 临时状态已保存 (缓存大小: {len(self.cached_data)})")
        except Exception as e:
            logger.error(f"保存临时状态失败: {e}")

    def _save_full_state(self):
        """
        【大步频保存】
        保存 FAISS，清空缓存，移动 ID 状态
        """
        try:
            logger.info("💾 正在执行全量持久化 (写入向量库)...")
            
            # 1. 保存 FAISS (最慢的一步)
            self.vec_manager.save_updated_store(self.vector_store, self.vector_store_path, self.index_name)
            
            # 2. 状态转移：Cache -> Committed
            # 因为数据已经进 FAISS 文件了，所以 cached_data 可以清空
            self.committed_ids.update(self.cached_data.keys())
            self.cached_data = {} # 清空内存缓存
            
            # 3. 清理/更新磁盘上的临时文件
            self._save_temp_state() # 这会把空的 cache 写入磁盘，并更新 committed_ids
            
            logger.info("✅ 全量保存完成，缓存已清空。")
        except Exception as e:
            logger.error(f"❌ 全量保存失败: {e}")

    def run(self, batch_size=4, small_step=100, big_step=2000):
        docstore = self.vec_manager.get_documents(self.vector_store)
        if not docstore:
            return

        # 计算待处理任务
        # 待处理 = 总文档 - (已入库 + 在缓存中)
        processed_ids = self.committed_ids.union(self.cached_data.keys())
        pending_items = [(k, v) for k, v in docstore.items() if k not in processed_ids]
        
        total = len(pending_items)
        if total == 0:
            logger.info("所有文档均已完成处理。")
            return

        logger.info(f"🚀 开始任务 | 待处理: {total} | 已入库: {len(self.committed_ids)} | 缓存中: {len(self.cached_data)}")
        
        counter = 0 # 仅用于本次运行的计数

        with tqdm(total=total, desc="NER处理中") as pbar:
            for i in range(0, total, batch_size):
                batch_items = pending_items[i : i + batch_size]
                
                texts = [doc.page_content for _, doc in batch_items]
                ids = [doc_id for doc_id, _ in batch_items]

                try:
                    # 1. LLM 推理
                    results = self.chain.batch([{"text": t} for t in texts])
                    
                    # 2. 内存更新 (VectorStore + CacheDict)
                    for doc_id, ner_result in zip(ids, results):
                        if ner_result:
                            # A. 更新到内存 VectorStore (为了检索能立刻用到，也为了最终 save)
                            docstore[doc_id].metadata["ner"] = ner_result
                            
                            # B. 更新到内存 CacheDict (为了小步频存盘)
                            self.cached_data[doc_id] = ner_result
                    
                    current_batch_len = len(batch_items)
                    counter += current_batch_len
                    pbar.update(current_batch_len)

                    # 3. 检查保存策略
                    
                    # 触发大步频 (落库)
                    if counter % big_step < batch_size and counter > 0:
                        self._save_full_state()
                    
                    # 触发小步频 (存缓存)
                    elif counter % small_step < batch_size:
                        self._save_temp_state()
                        # logger.info(f"🚀 开始任务 | 待处理: {total} | 已入库: {len(self.committed_ids)} | 缓存中: {len(self.cached_data)}")

                except Exception as e:
                    logger.error(f"Batch Error: {e}")

        # 循环结束后的最终保存
        self._save_full_state()


    def run_chuanliu(self, batch_size=1, small_step=100, big_step=2000):
            """
            改写后的单条处理模式
            batch_size 建议设为 1 以便精细化排查
            """
            docstore = self.vec_manager.get_documents(self.vector_store)
            if not docstore:
                return

            # 1. 计算待处理任务
            processed_ids = self.committed_ids.union(self.cached_data.keys())
            pending_items = [(k, v) for k, v in docstore.items() if k not in processed_ids]
            
            total = len(pending_items)
            if total == 0:
                logger.info("所有文档均已完成处理。")
                return

            # 2. 准备失败记录文件
            failed_log_path = os.path.join(Config.DOCS_DIR, "ner_failed_ids.txt")

            logger.info(f"🚀 串行模式启动 | 待处理: {total} | 已入库: {len(self.committed_ids)} | 缓存中: {len(self.cached_data)}")
            
            counter = 0 

            with tqdm(total=total, desc="NER处理中") as pbar:
                # 注意：即便这里传了 batch_size > 1，内部也会逐条处理以确保安全
                for i in range(0, total, batch_size):
                    batch_items = pending_items[i : i + batch_size]
                    
                    for doc_id, doc in batch_items:
                        text = doc.page_content
                        try:
                            # --- 核心修改：单条推理 ---
                            # 如果在 ChatOllama 初始化时设置了 timeout，这里会生效
                            ner_result = self.chain.invoke({"text": text})
                            
                            if ner_result:
                                # A. 更新内存向量库
                                docstore[doc_id].metadata["ner"] = ner_result
                                # B. 更新临时缓存字典
                                self.cached_data[doc_id] = ner_result
                            else:
                                logger.warning(f"⚠️ ID: {doc_id} 返回结果为空")

                        except Exception as e:
                            # --- 核心修改：记录失败 ID ---
                            logger.error(f"❌ 处理失败 | ID: {doc_id} | 错误: {e}")
                            with open(failed_log_path, "a", encoding="utf-8") as f:
                                f.write(f"{doc_id}\n")
                            # 失败后继续下一条，不中断程序
                            continue

                        finally:
                            counter += 1
                            pbar.update(1)

                        # 3. 检查保存策略 (移动到单条循环内，保证步频准确)
                        # 触发大步频 (落库 FAISS)
                        if counter > 0 and counter % big_step == 0:
                            self._save_full_state()
                            logger.info(f"💾 已完成第 {counter} 条的大步频全量保存")
                        
                        # 触发小步频 (存 JSON 缓存)
                        elif counter > 0 and counter % small_step == 0:
                            self._save_temp_state()

            # 循环彻底结束后的最终保存
            self._save_full_state()
            logger.info("🏁 所有任务处理完毕。")

if __name__ == "__main__":
    ner = NERProcessor(Config.VECTOR_STORE_PATH)
    # 小步频 100 存一次 json，大步频 2000 存一次 FAISS
    ner.run(batch_size=4, small_step=100, big_step=2000)