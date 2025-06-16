import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
import csv
import psutil
import GPUtil
import torch
import time
from pynvml import (nvmlInit,
                    nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetComputeRunningProcesses,
                    nvmlDeviceGetTemperature,
                    nvmlDeviceGetPowerUsage,
                    NVML_TEMPERATURE_GPU
                    )
import os
import gc
import subprocess
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

import time
from transformers import TrainerCallback
import csv

class TimeLoggingCallback(TrainerCallback):
    def __init__(self):
      self.start_time = None
      self.cpu_usage_before = None
      self.ram_usage_before = None
      self.ram_usage_unique_before = None

      self.torch_memory_reserved_before = None
      self.torch_memory_allocated_before = None
      self.nvml_gpu_memory_used_before = None
      self.nvidia_smi_gpu_usage_before = None

      self.power_before = None
      self.temp_before = None

      self.end_time = None
      self.cpu_usage_after = None
      self.ram_usage_after = None
      self.ram_usage_unique_after = None

      self.torch_memory_reserved_after = None
      self.torch_memory_allocated_after = None
      self.nvml_gpu_memory_used_after = None
      self.nvidia_smi_gpu_usage_after = None

      self.power_after = None
      self.temp_after = None

      self.process = psutil.Process()
      self.process.cpu_percent() # returns meaningless 0 first time it is called

    def on_step_begin(self, args, state, control, **kwargs):
      self.start_time = time.perf_counter()
      self.cpu_usage_before = get_cpu_usage(process=self.process)
      self.ram_usage_before = get_ram_usage(process=self.process)
      self.ram_usage_unique_before = get_unique_ram_usage(process=self.process)

      self.torch_memory_reserved_before = torch.cuda.memory_reserved() / (1024 * 1024)
      self.torch_memory_allocated_before = torch.cuda.memory_allocated() / (1024 * 1024)
      self.nvml_gpu_memory_used_before = nvml_gpu_usage_per_process(process=self.process)
      self.nvidia_smi_gpu_usage_before = record_nvidia_smi_gpu_usage_output()

      self.power_before, self.temp_before = get_power_and_temperature()

    def on_step_end(self, args, state, control, logs=None, **kwargs):
      self.end_time = time.perf_counter()
      self.cpu_usage_after = get_cpu_usage(process=self.process)
      self.ram_usage_after = get_ram_usage(process=self.process)
      self.ram_usage_unique_after = get_unique_ram_usage(process=self.process)

      self.torch_memory_reserved_after = torch.cuda.memory_reserved() / (1024 * 1024)
      self.torch_memory_allocated_after = torch.cuda.memory_allocated() / (1024 * 1024)
      self.nvml_gpu_memory_used_after = nvml_gpu_usage_per_process(process=self.process)
      self.nvidia_smi_gpu_usage_after = record_nvidia_smi_gpu_usage_output()

      self.power_after, self.temp_after = get_power_and_temperature()

      print(f"Step {state.global_step}: {self.end_time - self.start_time:.4f} seconds")
      
      write_data(file_name, [state.global_step, self.start_time, self.end_time, self.cpu_usage_before, self.cpu_usage_after, self.ram_usage_before, self.ram_usage_after, self.ram_usage_unique_before, self.ram_usage_unique_after, self.torch_memory_reserved_before, self.torch_memory_reserved_after, self.torch_memory_allocated_before, self.torch_memory_allocated_after, self.nvidia_smi_gpu_usage_before, self.nvidia_smi_gpu_usage_after, self.nvidia_smi_gpu_usage_before, self.nvidia_smi_gpu_usage_after, self.power_before, self.power_after, self.temp_before, self.temp_after])


def run():
  global file_name, N, n, i

  write_data(file_name, [], ["step (n)", "time_before (s)", "time_after (s)", "cpu_used_before (%)", "cpu_used_after (%)", "ram_used_before: RSS (mb)", "ram_used_after: RSS (mb)", "ram_used_before: USS (mb)", "ram_used_after: USS (mb)", "torch_memory_reserved_before (mb)", "torch_memory_reserved_after (mb)", "torch_memory_allocated_before (mb)", "torch_memory_allocated_after (mb)", "nvml_gpu_memory_used_before (mb)", "nvml_gpu_memory_used_after (mb)", "nvidia_smi_gpu_usage_before (mb)", "nvidia_smi_gpu_usage_after (mb)", "power_before (W)", "power_after (W)", "temp_before (C)", "temp_after (C)"])

  tr_dataset = dataset.select(range(i))

  for j in range(N):
    
    dpo_trainer = DPOTrainer(
      model=model,
      args=training_args,
      train_dataset=tr_dataset,
      processing_class=tokenizer,
      peft_config=peft_config,
      callbacks=[TimeLoggingCallback()]
    )

    dpo_trainer.train()

    # just a blank line for separation (did i spell that right?)
    write_data(file_name, [])

    del dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(1)

  del tr_dataset
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  time.sleep(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running DPO Benchmark")
    parser.add_argument("N", type=int, help="Number of Repititions")
    parser.add_argument("n", type=int, help="Dataset size (in powers of 2)")
    parser.add_argument("--use_f16", action="store_true", help="Use f16")

    args = parser.parse_args()

    N = args.N
    n = args.n
    use_f16 = args.use_f16
    
    dtype = None
    if use_f16:
       dtype = torch.float16
    else: dtype = torch.float32

    i = 2**n
    file_name = f"dpo_finetuning_dataset_{i}.csv"

    model_name = "meta-llama/Llama-2-7b-chat-hf"

    # load your dataset of choice
    dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load model in 4 bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map={"": 0},
      torch_dtype=dtype
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
      lora_alpha=256,
      lora_dropout=0.1,
      r=8, # default set to 8
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
      ],
    )

    training_args = DPOConfig(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=100,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=5e-7,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        do_eval=False,
        adam_epsilon=1e-08,
        save_strategy="no",
        output_dir=f"./store",
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        beta = 2,
        push_to_hub=False, # if you want to push to hugging face,
        label_names=["prompt", "chosen", "rejected"]
    )


    experiment_start_time = time.perf_counter()

    run()

    print(f"Experiment took {time.perf_counter() - experiment_start_time} seconds with N = {N}")

    exit()

