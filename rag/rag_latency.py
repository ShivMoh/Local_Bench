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

def write_data(file_name, data, headers = None) -> None:
  try:
    with open(file_name, mode='a', encoding='utf-8') as file:
      writer = csv.writer(file)
      if headers is not None:
        writer.writerow(headers)
      if len(data) > 0:
        writer.writerow(data)
      else:
         writer.writerow([]) # ik i could just do this above in one step but welll.....leave me alone.....
  except Exception as e:
    print(f"Error {e}")
## get utils

file_name = "CPU_Stats.csv"

write_data(file_name, [], ["CPU_Count", "CPU_Count (actual)", "CPU_Stats", "CPU_Load_Avg", "CPU_Frequency"])

write_data(file_name, [psutil.cpu_count(), psutil.cpu_count(logical=False), psutil.cpu_stats(), psutil.getloadavg(), psutil.cpu_freq(percpu=True)])

file_name = "RAM_Stats.csv"
write_data(file_name, [], ["Virtual_Memory", "Swap Memory"])
write_data(file_name, [psutil.virtual_memory(), psutil.swap_memory()])