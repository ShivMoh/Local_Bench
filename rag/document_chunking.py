import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List
import csv
import time
from pynvml import (nvmlInit,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetComputeRunningProcesses,
                    nvmlDeviceGetTemperature,
                    nvmlDeviceGetPowerUsage,
                    NVML_TEMPERATURE_GPU
                    )
import subprocess
import psutil
import torch
import argparse

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


huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

def total_length_of_documents(chunks : List[str]) -> int:
  return sum(len(doc.page_content) for doc in chunks)

# constant variables
chunk_size = 700
chunk_overlap = 50

# also total document length


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
    global N, file_name

    # create the file headers
    write_data(file_name, [], ["Run_#", "Time_Taken (s)", "Total_Length (n)", "Number_Of_Documents (l)", "cpu_used_before (%)", "cpu_used_after (%)", "ram_used_before: RSS (mb)", "ram_used_after: RSS (mb)", "ram_used_before: USS (mb)", "ram_used_after: USS (mb)", "torch_memory_reserved_before (mb)", "torch_memory_reserved_after (mb)", "torch_memory_allocated_before (mb)", "torch_memory_allocated_after (mb)", "nvml_gpu_memory_used_before (mb)", "nvml_gpu_memory_used_after (mb)", "nvidia_smi_gpu_usage_before (mb)", "nvidia_smi_gpu_usage_after (mb)", "power_before (W)", "power_after (W)", "temp_before (C)", "temp_after (C)"])

    nvmlInit()
    process = psutil.Process()
    process.cpu_percent()

    for sub in sub_folders:

        sub_folder = sub
        folder_path = f"./data/{sub_folder}"

        for i in range(0, N):
            start = time.perf_counter()
            gpu_usage_before = nvml_gpu_usage_per_process(process)
            cpu_usage_before = get_cpu_usage(process)
            ram_usage_before = get_ram_usage(process)
            ram_usage_unique_before = get_unique_ram_usage(process)
            torch_memory_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)
            torch_memory_allocated_before = torch.cuda.memory_allocated() / (1024 * 1024)
            power_before, temp_before = get_power_and_temperature()
            nvidia_smi_gpu_usage_before = record_nvidia_smi_gpu_usage_output()
            
            loader = PyPDFDirectoryLoader(folder_path)

            docs_before_split = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap  = chunk_overlap,
            )

            docs_after_split = text_splitter.split_documents(docs_before_split)

            gpu_usage_after = nvml_gpu_usage_per_process(process)
            cpu_usage_after = get_cpu_usage(process)
            ram_usage_after = get_ram_usage(process)
            ram_usage_unique_after = get_unique_ram_usage(process)
            torch_memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
            torch_memory_allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
            power_after, temp_after = get_power_and_temperature()
            nvidia_smi_gpu_usage_after = record_nvidia_smi_gpu_usage_output()
            end = time.perf_counter()

            time_per_run = end - start

            write_data(file_name, [i, time_per_run, sum([len(doc.page_content) for doc in docs_after_split]), len(docs_after_split), cpu_usage_before, cpu_usage_after, ram_usage_before, ram_usage_after, ram_usage_unique_before, ram_usage_unique_after, torch_memory_reserved_before, torch_memory_reserved_after, torch_memory_allocated_before, torch_memory_allocated_after, gpu_usage_before, gpu_usage_after, nvidia_smi_gpu_usage_before, nvidia_smi_gpu_usage_after, power_before, power_after, temp_before, temp_after])

            # basically free()
            torch.cuda.empty_cache()
            
        # print(sum([len(doc.page_content) for doc in docs_after_split]))

        write_data(file_name, [], ["","",""])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running DPO Benchmark")
    parser.add_argument("N", type=int, help="Number of Repititions")
   
    args = parser.parse_args()

    N = args.N
    file_name = "document_chunking.csv"
    sub_folders = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    sub_folder = "one"
    folder_path = f"./data/{sub_folder}"

    experiment_start_time = time.perf_counter()

    run()

    print(f"Experiment took {time.perf_counter() - experiment_start_time} seconds with N = {N}")

    exit()

