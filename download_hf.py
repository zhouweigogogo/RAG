from huggingface_hub import login, snapshot_download
import os

# 设置环境变量（如果需要）
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 登录（根据需要启用）
# login(token='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', add_to_git_credential=True)

# 强制下载数据集 qgyd2021/chinese_ner_sft
dataset_name = "qgyd2021/chinese_ner_sft"
local_dir = "/home/zhouw/study/RAG/tmp"  # 指定保存位置

try:
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=local_dir,
        force_download=True,  # 强制下载
        resume_download=False,  # 不续传，重新开始下载
        local_files_only=False  # 允许从网络下载文件
    )
    print(f"Dataset {dataset_name} has been downloaded successfully to {local_dir}.")
except Exception as e:
    print(f"An error occurred while downloading the dataset: {e}")