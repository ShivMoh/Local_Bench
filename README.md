## Overview
A benchmark created as a requirement for conducting inference benchmarks for a final year research project titled "Local and Personalized Large Language Models in Intelligent Tutoring Systems" conducted at the University of Guyana towards the Bachelor's in Computer Science

## Create conda environment 
```
conda create -n <name_of_environment> python=3.12
```

Activate using:
```
conda activate <name_of_environment>
```

See further instructions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
Install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

## Create venv environment

```
python3 -m venv path/to/environment
```

Activate using:
```
source path/to/environment/bin/activate
```

## Installation 
```python
pip install -U transformers trl peft optimum bitsandbytes numpy datasets sentence_transformers langchain faiss-cpu langchain_huggingface langchain_community pypdf sympy GPUtil psutil pynvml
```

## Execution
```bash
chmod u+x ./model.sh
nohup ./model.sh
```
_Note: See model.sh for more examples on how to run specific scripts. Note that for more preciseness, it may be necessary to edit the scripts themselves_

## Important
Benchmark only works for Nvidia GPUs. Dependecies on libs such as pynvml which utilises nvml prevent it from working with other GPU types. To run on other types of GPUs, remove pynvml from the scripts and utilise alternatives primarily sucha as GPUtil. However, certain metrics
such as power consumption may be not be possible for data collection due to a lack of alternatives.


## References
- Benchmark Work | Benchmarks MLCommons. (2025, February 8). MLCommons. https://mlcommons.org/benchmarks/
- Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C.-J., Anderson, B., Breughe, M., Charlebois, M., Chou, W., Chukka, R., Coleman, C., Davis, S., Deng, P., Diamos, G., Duke, J., Fick, D., Gardner, J. S., Hubara, I., & Idgunji, S. (2020, May 1). MLPerf Inference Benchmark. IEEE Xplore. https://doi.org/10.1109/ISCA45697.2020.00045
- Chitty-Venkata, K. T., Siddhisanket Raskar, Kale, B., Ferdaus, F., Aditya Tanikanti, Raffenetti, K., Taylor, V., Murali Emani, & Vishwanath, V. (2024). LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators. SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, 1362–1379. https://doi.org/10.1109/scw63240.2024.00178
- Lukas Tuggener, Sager, P., Yassine Taoudi-Benchekroun, Grewe, B. F., & Stadelmann, T. (2024). So you want your private LLM at home? A survey and benchmark of methods for efficient GPTs. 2024 11th IEEE Swiss Conference on Data Science (SDS), 205–212. https://doi.org/10.1109/sds60720.2024.00036
