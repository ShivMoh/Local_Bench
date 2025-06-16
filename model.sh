echo "Running rag latency experiment..."
nohup python ./rag/rag_latency.py 31 > rag.txt
sleep 5

echo "Running document chunking..."
nohup python ./rag/document_chunking.py 31 > chunk.txt
sleep 5

echo "Running bitsandbytes quantisation experiment..."
nohup python ./inference/inference_quantisation.py 1100 4 BNB --use_f16 > bnb.txt
sleep 5
nohup python ./load/model_load.py 31 BNB --use_f16 > bnb.txt
sleep 5 

echo "Installing GPTQ dependencies..."
nohup pip install gptqmodel --no-build-isolation
nohup python ./inference/inference_quantisation.py 1100 4 GPTQ --use_f16 > gptq.txt
sleep 5
nohup python ./load/model_load.py 31 GPTQ --use_f16 > bnb.txt
sleep 5

echo "Running DPO experiment..."
nohup python ./dpo/dpo_experiment.py 1 10 --use_f16 > dpo.txt
sleep 5  

echo "Running encoding and decoding..."
nohup python ./inference/encoding.py 31 4 > encoding.txt
nohup python ./inference/decoding.py 31 4 > decoding.txt
sleep 5

echo "Running AWQ experiment..."
nohup pip install autoawq autoawq-kernels
sleep 5

nohup python ./inference/inference_quantisation.py 1100 4 AWQ --use_f16 > awq.txt
nohup python ./load/model_load.py 31 AWQ --use_f16 > awq.txt

nohup python ./stats/cpu_stats.py
nohup python ./stats/spec.py

echo "We are done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"