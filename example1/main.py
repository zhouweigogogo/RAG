import faiss
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import pickle
import os

def get_ans(prompt):

    client = ZhipuAI(api_key="2cbd2e7e65ab3bff354ca6e27ba07514.euHhzQ3jynxFgtXI")

    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "system",
                "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。" 
            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ],
        top_p= 0.7,
        temperature= 0.95,
        max_tokens=1024,
        tools = [{"type":"function","function":{"name":"get_weather","description":"获取指定地点当前的天气情况","parameters":{"type":"object","properties":{"location":{"type":"string","description":"城市，例如：北京、上海"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}}}],
        stream=True
    )
    print(''.join([i.choices[0].delta.content for i in response]))
    # for trunk in response:
    #     print(trunk.content)


if __name__ == "__main__":
    if os.path.exists('./document'):

        loader = DirectoryLoader('./document')
        documents = loader.load()

        text_spliter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        split_doc = text_spliter.split_documents(documents=documents)
        contents = [i.page_content for i in split_doc]

        with open('contents.pkl','wb') as f:
            pickle.dump(contents, f)
    else:
        with open('contents.pkl','rb') as f:
            contents = pickle.load(f)

    sentence_model = SentenceTransformer('./Yuan-embedding-1.0')

    if not os.path.exists('faiss.pkl'):
        
        faiss_index = faiss.IndexFlatL2(sentence_model.get_sentence_embedding_dimension())
        faiss_index.add(sentence_model.encode(contents))

        with open('faiss.pkl','wb') as f:
            pickle.dump(faiss_index,f)
    else:
        with open('faiss.pkl','rb') as f:
            faiss_index = pickle.load(f)

    print("向量构建 结束")

    user_query = '什么是大模型'
    dim, index = faiss_index.search(sentence_model.encode([user_query]), 1)

    print("向量库检索内容：", contents[index[0][0]])

    prompt = f"请根据已知内容回复用户问题，已知内容如下：{contents[index[0][0]]}。用户问题如下：{user_query}。如果用户问题与已知内容完全不相关，请直接回复：不知道"

    get_ans(prompt)

