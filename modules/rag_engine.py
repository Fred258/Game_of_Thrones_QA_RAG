# -*- coding: utf-8 -*-
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from config import Config
from modules.prompts import RAGPrompts
from pydantic import SecretStr
from loguru import logger

class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        # 设置缓存
        set_llm_cache(InMemoryCache())
        
        # 初始化各功能模型
        # 1. 查询处理模型
        query_rewrite_model_api_key = Config.QUERY_REWRITE_MODEL_API_KEY
        self.query_rewrite_llm = ChatOpenAI(
            model=Config.QUERY_REWRITE_MODEL_NAME,
            temperature=Config.QUERY_REWRITE_MODEL_TEMPERATURE,
            api_key=SecretStr(query_rewrite_model_api_key) if query_rewrite_model_api_key else None,
            base_url=Config.QUERY_REWRITE_MODEL_BASE_URL
        )
        logger.info(f"查询改写模型:{Config.QUERY_REWRITE_MODEL_NAME}")
        # 2. 回答生成模型
        response_model_api_key = Config.RESPONSE_MODEL_API_KEY
        self.response_llm = ChatOpenAI(
            model=Config.RESPONSE_MODEL_NAME,
            temperature=Config.RESPONSE_MODEL_TEMPERATURE,
            api_key=SecretStr(response_model_api_key) if response_model_api_key else None,
            base_url=Config.RESPONSE_MODEL_BASE_URL
        )
        logger.info(f"回答生成模型:{Config.RESPONSE_MODEL_NAME}")

        # 3. 解析模型回答中的Json结构
        self.ner_parser = JsonOutputParser()

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        利用 LLM 从 Query 中提取实体，用于构建 Filter
        """
        prompt = PromptTemplate(
            template=RAGPrompts.QUERY_NER_TEMPLATE,
            input_variables=["text"]
        )
        # 打通 Prompt -> LLM -> Parser 流程
        chain = prompt | self.query_rewrite_llm | self.ner_parser
        try:
            logger.info("🔍 正在分析查询问题中的实体...")
            result = chain.invoke({"text": query})
            # 简单清洗，确保 key 存在
            cleaned_result = {k: result.get(k, []) for k in ["people", "locations", "times"]}
            logger.info(f"提取实体结果: {cleaned_result}, 系统将使用该结果进行实体过滤")
            return cleaned_result
        except Exception as e:
            logger.warning(f"实体提取失败，将降级为无过滤检索: {e}")
            return {}

    def _create_dynamic_filter(self, entities: Dict[str, List[str]]):
            """
            构建 FAISS 过滤逻辑
            """
            if not any(entities.values()):
                logger.error("未提取到任何实体，无法构建动态过滤器。")
                return None

            # 一个接收待过滤metadata参数，相当于书写过滤成功与否的逻辑
            def metadata_filter(metadata: Dict[str, Any]) -> bool:
                # 1. 检查文档是否有 NER 数据
                doc_ner = metadata.get("ner", {})
                if not doc_ner:
                    return False
                
                # 2. 逻辑匹配：只要 Query 里的任意一个实体出现在文档的 NER 列表中，即视为匹配
                # 这里采用“宽松匹配”策略，也可以改为“严格匹配”
                for label, values in entities.items():
                    if not values:
                        continue
                    doc_values = doc_ner.get(label, [])
                    # 检查两个列表是否有交集
                    if set(values) & set(doc_values):
                        return True
                
                return False
                
            return metadata_filter

    def _get_retriever(self, search_type: str = Config.SEARCH_TYPE, use_ner_filter:bool = False):
        """
        构建 MultiQuery 检索器
        Args:
            search_type: 检索类型，默认为 "mmr",还可选 "similarity"
            use_ner_filter: 是否开启实体过滤
        """
        base_kwargs = {
            "k": Config.RETRIEVER_K,
            "fetch_k": Config.RETRIEVER_FETCH_K
        }

        # 如果不使用过滤，直接返回标准的 MultiQueryRetriever
        if not use_ner_filter:
            base_retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=base_kwargs
            )
            return self._wrap_multi_query(base_retriever)
        
        # 若开启过滤，需要构建带过滤的动态检索器
        # 使用 RunnableLambda 包装检索过程，使其可以访问 runtime 的输入(query)
        # def retrieval_func(input_dict: Dict[str, Any]):
        #     question = input_dict["input"]
        #     # 1. 提取实体
        #     entities = self._extract_entities(question)
        #     # 2. 构建过滤器
        #     faiss_filter = self._create_dynamic_filter(entities)
        #     # 3. 动态配置 Retriever
        #     # 注意：FAISS 的 as_retriever 生成的对象如果再次修改 search_kwargs 可能会有深拷贝问题
        #     # 所以我们在这一步动态生成一个新的 retriever
        #     current_kwargs = base_kwargs.copy()
        #     if faiss_filter:
        #         current_kwargs["filter"] = faiss_filter  # type: ignore
        #         logger.info("✅ 已应用实体过滤器")
        #     else:
        #         logger.info("⚠️ 未提取到有效实体，跳过过滤")

        #     dynamic_retriever = self.vector_store.as_retriever(
        #         search_type=search_type,
        #         search_kwargs=current_kwargs
        #     )
            
        #     # 4. 执行检索 (这里依然可以套用 MultiQuery，但为了性能和逻辑清晰，建议先单次检索)
        #     # 如果非常需要 MultiQuery + Filter，需要将 Filter 传递给 MultiQuery 内部的 retriever
        #     # 简单起见，这里演示直接检索
        #     return dynamic_retriever.invoke(question)
        
        # # RunnableLambda将一般Python函数封装为可集成进Chain的专用类
        # # 可以使用LangChain中的.invoke()等方法，用 | 符号与其他管道接通
        # return RunnableLambda(retrieval_func)
    
        # --- 核心逻辑：对每一条改写后的查询进行 NER 提取 ---
        def multi_query_ner_flow(input_dict: Dict[str, Any]):
            original_query = input_dict["input"]
            
            # 1. 显式调用改写逻辑，获取多个子查询
            # 使用我们定义的 query_rewrite_llm 和 prompt
            rewrite_prompt = PromptTemplate(
                template=RAGPrompts.QUERY_REWRITE_TEMPLATE,
                input_variables=["question"]
            )
            # 这里我们手动通过 LLM 获取改写列表（假设 Prompt 要求换行分隔）
            rewrite_chain = rewrite_prompt | self.query_rewrite_llm
            rewrite_output = rewrite_chain.invoke({"question": original_query})
            
            # 解析改写后的问题列表 (处理字符串，去除空行)
            # 建议在 Prompt 中明确要求输出格式，此处假设按行分隔
            rewritten_queries = [original_query]  # 总是包含原始问题
            try:
                if hasattr(rewrite_output, 'content'):
                    lines = rewrite_output.content.strip().split("\n")
                    rewritten_queries.extend([line.strip() for line in lines if line.strip()])
            except Exception as e:
                logger.warning(f"改写模块出现异常,请排查问题,暂退回至原回答查询")
                rewritten_queries.extend(original_query)
            
            logger.info(f"🔄 最终共有 {len(rewritten_queries)} 条查询语句,分别为")
            for idx, query in enumerate(rewritten_queries):
                logger.info(f"第{idx+1}条查询语句：{query}")

            # 2. 对每一条查询执行：提取实体 -> 构建过滤 -> 执行检索
            all_documents = []
            seen_doc_ids = set()

            for idx, q in enumerate(rewritten_queries):
                logger.info(f"处理子查询 [{idx+1}]: {q}")
                
                # 为当前子查询提取实体
                entities = self._extract_entities(q)
                faiss_filter = self._create_dynamic_filter(entities)
                
                # 配置带过滤的检索参数
                current_kwargs = base_kwargs.copy()
                if faiss_filter:
                    current_kwargs["filter"] = faiss_filter
                
                # 执行单次检索
                # 直接调用 vector_store 的检索方法，效率更高
                if search_type == "mmr":
                    docs = self.vector_store.max_marginal_relevance_search(
                        q, **current_kwargs
                    )
                else:
                    docs = self.vector_store.similarity_search(
                        q, **current_kwargs
                    )
                
                # 3. 合并结果并去重（基于文档内容或 ID）
                for doc in docs:
                    # 使用 page_content 的 hash 或 metadata 中的 id 作为去重键
                    doc_id = hash(doc.page_content) 
                    if doc_id not in seen_doc_ids:
                        all_documents.append(doc)
                        seen_doc_ids.add(doc_id)

            logger.info(f"✅ 最终召回去重文档数: {len(all_documents)}")
            return all_documents

        return RunnableLambda(multi_query_ner_flow)

    def _wrap_multi_query(self, base_retriever):
        """
        封装 MultiQuery 逻辑，即不使用NER过滤
        """
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=RAGPrompts.QUERY_REWRITE_TEMPLATE
        )
        logger.info("构建 MultiQueryRetriever...")
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.query_rewrite_llm,
            prompt=query_prompt,
        )


    # def build_chain(self, use_ner_filter:bool = False):
        
    #     """
    #     构建完整的 RAG Chain，包含检索器、文档接口和回答生成链
    #     """
        
    #     # 定义好问答模型prompt
    #     qa_prompt = ChatPromptTemplate.from_messages([
    #         ("system", RAGPrompts.QA_SYSTEM_PROMPT),
    #         ("human", "{input}"),
    #     ])

    #     # 规范化参考文档格式
    #     document_prompt = PromptTemplate(
    #         input_variables=["page_content", "index"], 
    #         template="【文档编号:{index}】\n内容:{page_content}"
    #     )

    #     # 在构建 combine_docs_chain 时，需要对传入的 docs 进行预处理（增加 index 字段）
    #     # 这是一个简单的 RunnableLambda 处理逻辑
    #     def format_docs_with_index(input_dict):
    #         docs = input_dict["context"]
    #         for i, doc in enumerate(docs):
    #             doc.metadata["index"] = i + 1
    #         return input_dict

        
    #     # 1. 将文档接口整合进回答生成链
    #     combine_docs_chain = create_stuff_documents_chain(self.response_llm, qa_prompt,document_prompt=document_prompt)
    #     # 2. 初始化检索器
    #     retriever = self._get_retriever(use_ner_filter=use_ner_filter)
    #     # 3. 最终 RAG 链
    #     return create_retrieval_chain(retriever, combine_docs_chain)

    def build_chain(self, use_ner_filter: bool = False):
        
        """
        构建完整的 RAG Chain，包含动态索引预处理
        """
        # 1. 定义问答提示词
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", RAGPrompts.QA_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        # 2. 定义单个文档在 Prompt 中的展现格式
        # 这里的 index 必须对应 metadata 中的键
        document_prompt = PromptTemplate(
            input_variables=["page_content", "index"], 
            template="【文档编号:{index}】\n内容:{page_content}"
        )

        # 3. 构建基础的文档整合链
        combine_docs_chain = create_stuff_documents_chain(
            self.response_llm, 
            qa_prompt, 
            document_prompt=document_prompt
        )

        # 4. 初始化检索器（根据你的逻辑可能是 RunnableLambda 或 MultiQuery）
        retriever = self._get_retriever(use_ner_filter=use_ner_filter)

        # 5. 定义文档索引预处理逻辑 (核心修改点)
        def add_index_to_docs(input_dict):
            # create_retrieval_chain 运行到这一步时，context 里已经是 List[Document]
            docs = input_dict["context"]
            for i, doc in enumerate(docs):
                # 将索引存入 metadata，这样 document_prompt 才能读取到
                doc.metadata["index"] = i + 1
            return input_dict

        # 6. 组装最终链条
        # 逻辑：检索 -> 添加索引 -> 送入文档整合链
        # 我们使用 create_retrieval_chain 作为基础，但通过 | 插入拦截逻辑
        base_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        # 使用 RunnableLambda 在数据流中进行“拦截并修改”
        final_chain = base_chain | RunnableLambda(add_index_to_docs) | combine_docs_chain
        
        # 注意：create_retrieval_chain 本身返回的是一个包含所有信息的字典
        # 为了保持接口统一，最优雅的写法是手动构建这个流：
        
        final_rag_chain = (
            # 第一步：检索并保留原始输入
            RunnablePassthrough.assign(context=retriever)
            # 第二步：给检索到的 context 添加 index 字段
            | RunnableLambda(add_index_to_docs)
            # 第三步：生成回答并保留 context
            | RunnablePassthrough.assign(answer=combine_docs_chain)
        )

        return final_rag_chain