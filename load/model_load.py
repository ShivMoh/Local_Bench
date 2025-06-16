import gc
import torch
import time
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
import argparse


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

def record_nvidia_smi_gpu_usage_output():
    # Run nvidia-smi command and capture output
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True
    )

    # this is perfect code and i will not hear anything
    mem_usage = output.split(",")[-1].replace(" ", "").replace("\n", "")

    return mem_usage


def get_cpu_usage(process):
    return process.cpu_percent() / psutil.cpu_count()

def get_ram_usage(process):
    return process.memory_info().rss / (1024 * 1024)  

def get_unique_ram_usage(process):
    return process.memory_full_info().uss / (1024 * 1024)  


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
  global N, flash_attention, quant_type, dtype
  
  quant_types = [quant_type]
  
  print("Flash attention is", flash_attention)
  print("Use f16 is", dtype)

  if flash_attention:
     config = "flash_attention"
  else: config = "no_flash_attention"
  
  process = psutil.Process()
  process.cpu_percent() # meaningless 0

  for quant_type in quant_types:
    
      file_name = f"model_load_{config}_{quant_type}_{str(dtype)}.csv"

      write_data(file_name, [], ["Run_#", "time_before (s)", "time_after (s)", "cpu_used_before (%)", "cpu_used_after (%)", "ram_used_before: RSS (mb)", "ram_used_after: RSS (mb)", "ram_used_before: USS (mb)", "ram_used_after: USS (mb)", "torch_memory_reserved_before (mb)", "torch_memory_reserved_after (mb)", "torch_memory_allocated_before (mb)", "torch_memory_allocated_after (mb)", "nvml_gpu_memory_used_before (mb)", "nvml_gpu_memory_used_after (mb)", "nvidia_smi_gpu_usage_before (mb)", "nvidia_smi_gpu_usage_after (mb)", "power_before (W)", "power_after (W)", "temp_before (C)", "temp_after (C)"])

      for i in range(0, N):

        cpu_usage_before = get_cpu_usage(process=process)
        ram_usage_before = get_ram_usage(process=process)
        ram_usage_unique_before = get_unique_ram_usage(process=process)

        torch_memory_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)
        torch_memory_allocated_before = torch.cuda.memory_allocated() / (1024 * 1024)
        nvml_gpu_memory_used_before = nvml_gpu_usage_per_process(process=process)
        nvidia_smi_gpu_usage_before = record_nvidia_smi_gpu_usage_output()

        power_before, temp_before = get_power_and_temperature()

        time_before = time.perf_counter()

        model = load_model(quantization_technique=quant_type, enabled_flash_attention=flash_attention, dtype=dtype)

        time_after = time.perf_counter()
        power_after, temp_after = get_power_and_temperature()
        cpu_usage_after = get_cpu_usage(process=process)
        ram_usage_unique_after = get_unique_ram_usage(process=process)
        ram_usage_after = get_ram_usage(process=process)
        
        torch_memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
        torch_memory_allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
        nvml_gpu_memory_used_after = nvml_gpu_usage_per_process(process=process)
        nvidia_smi_gpu_usage_after = record_nvidia_smi_gpu_usage_output()

        write_data(file_name, [i, time_before, time_after, cpu_usage_before, cpu_usage_after, ram_usage_before, ram_usage_after, ram_usage_unique_before, ram_usage_unique_after, torch_memory_reserved_before, torch_memory_reserved_after, torch_memory_allocated_before, torch_memory_allocated_after, nvml_gpu_memory_used_before, nvml_gpu_memory_used_after, nvidia_smi_gpu_usage_before, nvidia_smi_gpu_usage_after, power_before, power_after, temp_before, temp_after])

        # free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Wait for memory release
        time.sleep(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running DPO Benchmark")
    parser.add_argument("N", type=int, help="Number of Repititions")
    parser.add_argument("quant_type", type=str, help="Quantisation Type To Use")
    parser.add_argument("--flash_attention", action="store_true", help="Enable Flash Attention")
    parser.add_argument("--use_f16", action="store_true", help="Enable Flash Attention")

    args = parser.parse_args()

    N = args.N
    flash_attention = args.flash_attention
    quant_type = args.quant_type

    if args.use_f16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    experiment_start_time = time.perf_counter()

    run()

    print(f"Experiment took {time.perf_counter() - experiment_start_time} seconds with N = {N}")

    exit()

