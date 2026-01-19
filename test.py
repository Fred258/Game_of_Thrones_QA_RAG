# from langchain_community.document_loaders import TextLoader
# try:
#     path = r"D:/RAG/langchain_project/translatorAssistant/basic_knowledge/rag_test/docs/Game_of_Thrones/GameofThrones.txt"
#     loader = TextLoader(path, encoding='gbk')
#     doc = loader.load()
#     print("加载成功！")
# except Exception as e:
#     print(f"底层错误原因: {e}")


# file_path = r"D:/RAG/langchain_project/translatorAssistant/basic_knowledge/rag_test/docs/Game_of_Thrones/GameofThrones.txt"

# try:
#     with open(file_path, 'r', encoding='gbk') as f:
#         content = f.read(100) # 只读前100个字
#         print("【原生读取成功】内容预览:", content)
# except FileNotFoundError:
#     print("【系统报错】找不到文件，请检查路径是否拼写错误。")
# except UnicodeDecodeError:
#     print("【系统报错】编码错误！文件可能不是 UTF-8 编码，请尝试 encoding='gbk'。")
# except PermissionError:
#     print("【系统报错】权限拒绝，文件可能被占用或权限不足。")
# except Exception as e:
#     print(f"【系统报错】其他原因: {e}")

# from huggingface_hub import snapshot_download

# local_dir = r"D:/RAG/langchain_project/translatorAssistant/basic_knowledge/models/bge-large-zh-v1.5"

# snapshot_download(
#     repo_id="BAAI/bge-large-zh-v1.5",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # Windows 必须 False
#     resume_download=True
# )

# print("模型下载完成，路径：", local_dir)


# from huggingface_hub import snapshot_download

# local_dir = r"D:/RAG/langchain_project/translatorAssistant/basic_knowledge/models/bert-ner-chinese"

# snapshot_download(
#     repo_id="ckiplab/bert-base-chinese-ws-finetuned-ner_all",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # Windows 必须 False
#     resume_download=True
# )

# print("BERT-NER 模型下载完成：", local_dir)

# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# model_path = r"D:/RAG/langchain_project/translatorAssistant/basic_knowledge/models/bert-chinese-ner"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForTokenClassification.from_pretrained(model_path)

# ner = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )

# text = "我爱吃苹果"
# print(ner(text))

# import requests

# response = requests.post(
#     "http://localhost:11434/api/generate",
#     json={
#         "model": "qwen2.5:1.5b-instruct",
#         "prompt": "请抽取下面文本中的人物和事件：奈德·史塔克在贝勒大圣堂前被斩首。",
#         "stream": False
#     }
# )

# print(response.json()["response"])

from modules.rag_engine import RAGEngine

if __name__ == "__main__":
    rag = RAGEngine(None)
    entities = rag._extract_entities("请问奈德·史塔克在君临发生了什么事？")
    print(entities)