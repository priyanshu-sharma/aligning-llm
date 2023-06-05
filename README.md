# Aligning-LLM

Official Repo for Reinforcement Learning Project (Aligning LLM with Human Preception)

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

