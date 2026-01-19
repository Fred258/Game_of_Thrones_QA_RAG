# -*- coding: utf-8 -*-
import os
from loguru import logger
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    def __init__(self):
        
        """
        文档处理器初始化
        args:
            chunk_size: 切分块大小
            chunk_overlap: 切分块重叠大小
            separators: 切分符列表
        """
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.separators = Config.SEPARATORS
        self.file_processed = []
        self.file_fail_to_process = []
        self.file_not_processed = []

    def _load_single_file(self, file_path: str)-> list:
        
        """
        内部方法：加载单个文件，包含编码容错逻辑
        Args:
            file_path: 文件路径
        
        """
        if file_path.endswith('.txt'):
            try:
                return TextLoader(file_path, encoding='utf-8').load()
            except (UnicodeDecodeError, RuntimeError):
                logger.info(f"使用 UTF-8 读取失败，将尝试 GBK 编码: {os.path.basename(file_path)}")
                try:
                    return TextLoader(file_path, encoding='gbk').load()
                except Exception as e:
                    logger.error(f"加载失败: {file_path} - {e}")
                    return []
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path).load()
        elif file_path.endswith('.docx'):
            return Docx2txtLoader(file_path).load()
        else:
            return []

    def load_directory(self, directory_path: str, valid_exts:list=['.txt', '.pdf', '.docx']) -> list:
        
        """
        加载目录下所有支持的文档
        Args:
            directory_path: 目录路径
            valid_exts: 支持的文件扩展名列表
        """

        logger.info(f"正在读取待构建为检索库的文档目录: {directory_path}")
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"文档目录不存在: {directory_path}")
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in valid_exts):
                logger.info(f"   -> 载入文档: {filename}")
                try:
                    docs = self._load_single_file(file_path)
                    documents.extend(docs)
                    self.file_processed.append(filename)
                except Exception as e:
                    logger.warning(f"加载文档失败: {filename} - {e}")
                    self.file_fail_to_process.append(filename)
            else:
                self.file_not_processed.append(filename)

        logger.info(f"成功载入{self.file_processed}")
        logger.info(f"载入失败{self.file_fail_to_process}")
        logger.info(f"无需处理文件{self.file_not_processed}")
        logger.info(f"文档数据载入完成，共{len(documents)}个文档")

        return documents

    def split_documents(self, documents:list) -> list:
        
        """
        切分文档

        Args:
            documents: 已经成功读取后的文档内容列表
        """
        if len(documents) == 0:
            logger.warning("没有可切分的文档，返回空列表")
            return []
        logger.info("正在切分待检索文档...")
        logger.info(f"切分参数 - chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}, separators: {self.separators}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"待检索文档已全部切分为 {len(chunks)} 个块")
        
        return chunks