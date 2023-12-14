import torch
from transformers import AutoTokenizer, AutoModel

model_name = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, use_cache=True, torchscript=True, trust_remote_code=True).cuda(0)

def run_model(query):
    past_key_values, history = None, []
    current_length = 0
    response_str=''
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True, temperature=0.001):

            response_str += response[current_length:]
            current_length = len(response)
    return response_str

run_model("?")
