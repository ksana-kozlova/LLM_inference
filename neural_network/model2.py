from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc
import time

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def flush():
    gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()

system_prompt = """
Question: Write a function that takes two lists and returns a list that has alternating elements from each input list.
def alternating(list1, list2):
   results = []
   for i in range(len(list1)):
       results.append(list1[i])
       results.append(list2[i])
   return results
"""
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

long_prompt = 10 * system_prompt + prompt

def create_pipe():
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return pipe

def run_model(message):
    pipe = create_pipe()
    start_time = time.time()
    result = pipe(message, max_new_tokens=60)[0]["generated_text"][len(message):]
    return f"Generated in {time.time() - start_time} seconds."

if __name__=='__main__':

    pipe = create_pipe()

    t_ = run_model(pipe)
    
    print(t_)
    # print(result)

    # bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
    # flush()
    
    # _ = model.to_bettertransformer()
    
    # start_time = time.time()
    # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
    #     result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]

    # print(f"Generated in {time.time() - start_time} seconds.")
    # result
