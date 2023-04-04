import json
import openaifunc
import numpy as np
import re


class User:
    def __init__(self):
        self.question = "曾国藩与左宗棠、李鸿章、张之洞并称“晚清四大名臣”，官至武英殿大学士，世袭罔替。"
        self.texts = {}
        self.embedding = {}

    def get_context_embedding(self, context: str) -> list[list]:
        return openaifunc.openapi().get_text_embeddings(context)

    def paragraph_division(self) -> None:  # 将数据源进行按照段进行切分
        with open("../datastorage/source.txt", 'r', encoding='utf-8') as f:
            texts = re.split(r'\n\n|\n', f.read())  # 按照双换行符分割段落
        for i, text in enumerate(texts):
            self.texts[i] = text

    def make_text_embedding(self):  # 建立数据源进行向量映射
        embeddings = []
        for text in self.texts.values():
            embeddings.append(self.get_context_embedding(text)[0])
        # 存储 embedding 向量列表和对应关系
        embedding_dict = {}
        for i, embedding in enumerate(embeddings):
            embedding_dict[i] = embedding
        # 将 embedding_dict 字典保存到json文件
        with open("../datastorage/embedding_dict.json", "w") as f:
            json.dump(embedding_dict, f)
        # 将 embeddings 列表保存为npy文件
        np.save("../datastorage/embeddings.npy", embeddings)

    def make_question_embedding(self):
        question_embedding_dict = {}
        question_embedding = self.get_context_embedding(self.question);
        for i, embedding in enumerate(question_embedding):
            question_embedding_dict[i] = embedding
        # 将 embedding_dict 字典保存到json文件
        with open("../datastorage/question_embedding_dict.json", "w") as f:
            json.dump(question_embedding_dict, f)
        # 将 embeddings 列表保存为npy文件
        np.save("../datastorage/question_embedding.npy", question_embedding)

    def similarity_order(self):
        question_embedding = np.load("../datastorage/question_embedding.npy")
        embeddings = np.load("../datastorage/embeddings.npy")
        question_embedding = question_embedding.reshape(-1, 1)
        order = []
        for embedding in embeddings:
            order.append(self.calculate_similarity([embedding], [question_embedding]))

        # 对相似度进行排序，获取排序后的索引
        sorted_indices = np.argsort([sim[0][0][0] for sim in order])[::-1]

        # 将排序后的相似度和对应的索引保存在一个字典里
        similarity_dict = {i: order[i][0][0][0] for i in sorted_indices}
        # 输出字典中的内容
        print(similarity_dict)

        # 输出相似度排序
        print(self.question)
        for i in sorted_indices:
            print(self.texts[i])
            print(similarity_dict[i])

    def calculate_similarity(self, embedding, question_embedding):
        return np.dot(embedding, question_embedding) / (
                np.linalg.norm(embedding, axis=1) * np.linalg.norm(question_embedding))
