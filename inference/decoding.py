import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import os
import time
import psutil
import GPUtil
import torch
import csv
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from pynvml import (nvmlInit,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetComputeRunningProcesses,
                    nvmlDeviceGetTemperature,
                    nvmlDeviceGetPowerUsage,
                    NVML_TEMPERATURE_GPU
                    )
import subprocess
import os
import argparse
import numpy as np
from datasets import load_dataset

dataset = load_dataset("Open-Orca/OpenOrca", split='train')




LLAMA_2_7 = "meta-llama/Llama-2-7b-chat-hf"
LLAMA_2_7_AWQ = "TheBloke/Llama-2-7B-Chat-AWQ"
LLAMA_2_7_GPTQ = "TheBloke/Llama-2-7b-chat-GPTQ"

def load_model(quantization_technique = "BNB", enabled_flash_attention = False, dtype=torch.float16):
  model = None
  ## AWQ quantization
  if quantization_technique == "AWQ":
    if enabled_flash_attention:
      model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7_AWQ,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        torch_dtype=dtype
      )
    else:
       model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7_AWQ,
        device_map={"": 0},
        torch_dtype=dtype
      )
  ## GPTQ Quantization
  elif quantization_technique == "GPTQ":
    if enabled_flash_attention:
      model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7_GPTQ,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        revision="main",
        torch_dtype=dtype
      )
    else:
      model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7_GPTQ,
        device_map={"": 0},
        revision="main",
        torch_dtype=dtype
      )
  ## Bits and Bytes Quantization
  elif quantization_technique == "BNB":
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type="nf4",
      torch_dtype=dtype
    )

    if enabled_flash_attention:
      model = AutoModelForCausalLM.from_pretrained(
          LLAMA_2_7,
          quantization_config=bnb_config,
          device_map={"": 0},
          attn_implementation="flash_attention_2",
          torch_dtype=dtype
      )
    else:
      model = AutoModelForCausalLM.from_pretrained(
        LLAMA_2_7,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=dtype
      )
  # No quantiazation
  else:
    model = AutoModelForCausalLM.from_pretrained(
      LLAMA_2_7,
      device_map={"": 0},
      attn_implementation="flash_attention_2",
      torch_dtype=dtype
    )
  if model is None:
    assert("Something went wrong")

  return model


def get_gpu_temp():
    temp = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader").read().strip()
    return temp

def get_gpu_power():
    power = os.popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader").read().strip()
    return power

def record_nvidia_smi_gpu_usage_output():
    # Run nvidia-smi command and capture output
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True
    )

    # this is perfect code and i will not hear anything
    mem_usage = output.split(",")[-1].replace(" ", "").replace("\n", "")

    return mem_usage

def nvml_gpu_usage_per_process(process):

    nvmlInit()  
    handle = nvmlDeviceGetHandleByIndex(0)  
    
    processes = nvmlDeviceGetComputeRunningProcesses(handle)

    for process in processes:
        if process.pid == process.pid:
            return process.usedGpuMemory / (1024 * 1024)  # Convert bytes to MB

    return 0  # Return 0 if process not found

def get_power_and_temperature():
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    gpu_power = nvmlDeviceGetPowerUsage(handle) / 1000.0

    return [gpu_power, gpu_temp]

def get_cpu_usage(process):
    return process.cpu_percent() / psutil.cpu_count()

def get_ram_usage(process):
    return process.memory_info().rss / (1024 * 1024)  

def get_unique_ram_usage(process):
    return process.memory_full_info().uss / (1024 * 1024)  

# Define a function to calculate latency metrics
def measure(prompt, model, tokenizer, max_new_tokens=100, i=0, process = None, tokens = 250):
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  tokens_in_prompt = input_ids.shape[1] 
  
  # streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
  
  # if not os.path.exists("./gpu_usage"):
  #   os.makedirs("gpu_usage")
  
  # file_name = f"./gpu_usage/output_{i}.csv"

  # with open(file_name, "a") as file:
  #   file.write("Memory_Used_MB\n")  # CSV Header

  # command = f'watch -n 0.5 "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | tee -a {file_name}"'

  # sub_process = subprocess.Popen(
  #     ["bash", "-c", command],
  #     stdout=subprocess.DEVNULL, 
  #     stderr=subprocess.STDOUT,
  #     text=True
  # )

  cpu_usage_before = get_cpu_usage(process=process)
  ram_usage_before = get_ram_usage(process=process)
  ram_usage_unique_before = get_unique_ram_usage(process=process)

  torch_memory_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)
  torch_memory_allocated_before = torch.cuda.memory_allocated() / (1024 * 1024)
  nvml_gpu_memory_used_before = nvml_gpu_usage_per_process(process=process)
  nvidia_smi_gpu_usage_before = record_nvidia_smi_gpu_usage_output()

  encoded_tokens = tokenizer(prompt, return_tensors="pt")

  power_before, temp_before = get_power_and_temperature()

  ## checking how long it takes for the tokenizer to encode information
  tokenizer.decode(input_ids[0])

  start_time = time.perf_counter()
  
  time_taken = time.perf_counter() - start_time
  power_after, temp_after = get_power_and_temperature()
  cpu_usage_after = get_cpu_usage(process=process)
  ram_usage_unique_after = get_unique_ram_usage(process=process)
  ram_usage_after = get_ram_usage(process=process)
  
  torch_memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
  torch_memory_allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
  nvml_gpu_memory_used_after = nvml_gpu_usage_per_process(process=process)
  nvidia_smi_gpu_usage_after = record_nvidia_smi_gpu_usage_output()
  time.sleep(1)

  return [i, time_taken, tokens_in_prompt, cpu_usage_before, cpu_usage_after, ram_usage_before, ram_usage_after, ram_usage_unique_before, ram_usage_unique_after, torch_memory_reserved_before, torch_memory_reserved_after, torch_memory_allocated_before, torch_memory_allocated_after, nvml_gpu_memory_used_before, nvml_gpu_memory_used_after, nvidia_smi_gpu_usage_before, nvidia_smi_gpu_usage_after, power_before, power_after, temp_before, temp_after]

def write_data(file_name, data, headers = None) -> None:
  try:
    with open(file_name, mode='a', encoding='utf-8') as file:
      writer = csv.writer(file)
      if headers is not None:
        writer.writerow(headers)
      if len(data) > 0:
        writer.writerow(data)
  except Exception as e:
    print(f"Error {e}")

def run():
  global N, K
  REPITITION_COUNT = N
  quant_types = ["BNB"]
  flash_attention = True

  tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7, trust_remote_code=True)
  tokenizer.add_special_tokens({'pad_token': '<PAD>'})

  config = ("no_flash", "stream")

  if config[0] == "flash":
      flash_attention = True
  else: flash_attention = False

  nvmlInit()
  process = psutil.Process()

  for quant_type in quant_types:
    file_name = f"decoding_{config[0]}_{config[1]}_{quant_type}.csv"

    write_data(file_name, [], ["Run_#", "time_taken (s)", "tokens_in_prompt (n)", "cpu_used_before (%)", "cpu_used_after (%)", "ram_used_before: RSS (mb)", "ram_used_after: RSS (mb)", "ram_used_before: USS (mb)", "ram_used_after: USS (mb)", "torch_memory_reserved_before (mb)", "torch_memory_reserved_after (mb)", "torch_memory_allocated_before (mb)", "torch_memory_allocated_after (mb)", "nvml_gpu_memory_used_before (mb)", "nvml_gpu_memory_used_after (mb)", "nvidia_smi_gpu_usage_before (mb)", "nvidia_smi_gpu_usage_after (mb)", "power_before (W)", "power_after (W)", "temp_before (C)", "temp_after (C)"])

    
    # first time called returns a pointless 0 
    process.cpu_percent()

    model = None

    for i in range(REPITITION_COUNT):
      for n, data in enumerate(bench_dataset): 
          
          if len(data["question"]) <= 4096:
            metrics = measure(data["question"], i=n, model=model, tokenizer=tokenizer, process=process, tokens=250)
            write_data(file_name, metrics)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(1)
          else:
             print(f"Skipped {n}")

    print("Writing data")
    write_data(file_name, ["", "", "", "", "", "", "", "", "", "", "", "", ""])

    # delete the model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    time.sleep(1)
    
K = 0 # number of prompts to use
N = 0 # number of experiment trials

prompt_index = 0

if __name__ == "__main__":
    global bench_dataset

    parser = argparse.ArgumentParser(description="Running Inference Benchmark")
    parser.add_argument("K", type=int, help="Number of Prompts to Use")
    parser.add_argument("N", type=int, help="Number of Loop Iterations")

    args = parser.parse_args()

    K = args.K
    N = args.N

    bench_dataset = dataset.select(range(K))
    
    experiment_start_time = time.perf_counter()

    run()

    print(f"Experiment took {time.perf_counter() - experiment_start_time} seconds with N = {N}")

    exit()
