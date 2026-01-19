from modules.ner import NERProcessor
from config import Config


if __name__ == "__main__":
    ner = NERProcessor(Config.VECTOR_STORE_PATH, index_name="index")
    # 小步频 100 存一次 json，大步频 2000 存一次 FAISS
    ner.run_chuanliu(batch_size=1, small_step=100, big_step=2000)