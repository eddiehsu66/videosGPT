import openai
import tiktoken
import numpy as np
import json
from app import PrivacyDocument


class openapi():
    def __init__(self, session_id=None):
        self.api_key = PrivacyDocument.api_key
        self.messages = []
        self.session_id = session_id
        self.systemcontent = ""
        d = {"role": "system", "content": self.systemcontent}
        self.messages.append(d)

    def open_api(self) -> str:
        openai.api_key = self.api_key
        res = openai.ChatCompletion.create(
            # messages=[{"role": "user", "content": content}] 表示将用户的文本作为输入
            # temperature=0.5 表示生成的文本多样性程度
            # max_tokens=1000 表示最大生成文本的长度
            # top_p=1 表示用于控制生成概率的参数，
            # frequency_penalty=0 和 presence_penalty=0 分别表示频率和存在惩罚的参数
            # stop=None 表示生成的文本不受特定标记的限制
            # user=self.session_id 表示为当前对话设置唯一的 session id。
            model="gpt-3.5-turbo",
            max_tokens=1000,
            messages=self.messages,
            # 角色有:user,system,assistant
            temperature=0.5,
            user=self.session_id,
        )
        return res.choices[0].message.content

    def add_message(self, role: str, content: str) -> None:
        if role not in ['system', 'user', 'assistant']:
            print("未指定角色")
        content.replace(" ", "")
        d = {"role": role, "content": content}
        self.messages.append(d)

    def ask_example(self):
        text1 = "什么是向量化"
        text2 = "可以再深入说一下么"

        self.add_message("user", text1)
        print(text1)
        answer1 = self.open_api()
        print(answer1)
        self.add_message("assistant", answer1)

        self.add_message("user", text2)
        print(text2)
        answer2 = self.open_api()
        print(answer2)
        print(self.calculate_token(self.messages))

    def calculate_token(self, messages, model="gpt-3.5-turbo") -> int:
        encoding = tiktoken.encoding_for_model(model)
        tokens_per_message = 4
        tokens_per_name = -1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    # 定义一个函数，将文本切割成小块，并返回每个块的 embedding 向量
    def get_text_embeddings(self, text: str, max_length=50) -> list[list]:
        # 将文本切割成小块
        text_blocks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        embeddings = []
        openai.api_key = self.api_key
        # 对每个文本块进行编码，并将结果添加到 embeddings 列表中
        for block in text_blocks:
            response = openai.Embedding.create(input=block, model="text-embedding-ada-002")
            embeddings.append(response["data"][0]["embedding"])
        return embeddings

    def get_embedding(self, text="这是一段文本，需要进行编码。") -> None:
        # 调用 get_text_embeddings 函数获取文本的 embedding 向量列表
        embeddings = self.get_text_embeddings(text)

        # 存储 embedding 向量列表和对应关系
        embedding_dict = {}
        for i, embedding in enumerate(embeddings):
            embedding_dict[i] = embedding
        # 将 embedding_dict 字典保存到json文件
        with open("../datastorage/embedding_dict.json", "w") as f:
            json.dump(embedding_dict, f)
        # 将 embeddings 列表保存为npy文件
        np.save("../datastorage/embeddings.npy", embeddings)

    def whisper_api(self):
        audio_file = open("../datastorage/videos1.mp3", "rb")
        openai.api_key = self.api_key
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
