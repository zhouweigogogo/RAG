#模型下载
from modelscope import snapshot_download

# model_dir = snapshot_download('LLM-Research/Llama-3.2-7B-Instruct', cache_dir='/home/buding666/zhouw/models/llama3.2-3B-instruct')
# model_dir = snapshot_download('BAAI/bge-base-zh-v1.5', cache_dir='/home/shared/class/zhouw/llama-index/qwen/embedding')
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='/home/zhouw/study/RAG/example2/download_models/Qwen2.5-1.5B-Instruct')  # BAAI/bge-base-zh-v1.5