import torch
from transformers import AutoTokenizer, AutoModel

model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, use_cache=False, trust_remote_code=True).cuda(1)

def run_model(query):
    history = None
    current_length = 0
    response_str=''
    for response, history in model.stream_chat(tokenizer, query, history=history,temperature=0.001):
        response_str += response[current_length:]
        current_length = len(response)
    return response_str

run_model("?")
