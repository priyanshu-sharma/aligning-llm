# Aligning-LLM with Human Perception

Official Repo for Reinforcement Learning Project (Aligning LLM with Human Preception)


Priyanshu Sharma

862395994

## Submodule Details

Follows Microbackend Architecture and composed of following Submodules: -

1. Trlx - https://github.com/CarperAI/trlx.git
2. Model Domain - Composed of various experiements on Bert, Transformer and LLama Models

## Setup

1. Clone the Repo -

```
git clone --recursive https://github.com/priyanshu-sharma/aligning-llm.git
```

2. Add configuration to update submodules recursively

```
git submodule update --init --recursive
```

Source - https://dev.to/jjokah/submodules-a-git-repo-inside-a-git-repo-36l9

3. Create Conda Environment

```
conda create -n env_aligning_llm python=3.10
pip install -r requirements.txt
```

4. Overall uses Python 3.10.10 and install other dependencies

```
cd src/trlx
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
```

## Results

Other training related graphs and results are also available at - https://drive.google.com/drive/folders/1oIeO_jX9p2YDfOo9P2vj-W8ECId-hAf0?usp=sharing

### PPO

T5 PPO - https://wandb.ai/pshar053/Aligning-LLM/reports/Weave-samples-23-06-16-12-24-54---Vmlldzo0NjY2MzI1
GPT PPO - https://wandb.ai/pshar053/Aligning-LLM/reports/Weave-samples-23-06-16-12-57-22---Vmlldzo0NjY2NDYx
Llama PPO - https://wandb.ai/pshar053/Aligning-LLM/reports/Weave-samples-23-06-16-12-57-53---Vmlldzo0NjY2NDYz

### ILQL

T5 ILQL - https://wandb.ai/pshar053/Aligning-LLM/reports/Weave-samples-23-06-16-13-01-20---Vmlldzo0NjY2NDgy
GPT ILQL - https://wandb.ai/pshar053/Aligning-LLM/reports/Weave-samples-23-06-16-13-00-25---Vmlldzo0NjY2NDc5
Llama ILQL - Not Currently Supported by trlx library

## Issues

1. ILQL Method for Llama Model is not working as it is currently not supported by trlx library. (src/model/ilql/llama.py)