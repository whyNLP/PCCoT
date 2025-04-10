# Huggingface Transformers Starter Code

General starter code for creative model architecture with huggingface transformer library. Users may implement customized llama model in `models` directory.

## Installation

Change the cuda version if it is not compatible. Developped with python 3.12.4.

```bash
conda install pytorch=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install mkl=2022.1.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -r requirements.txt
```

## Usage

Copy `run_clm.sh.template` to `run_clm.sh` and run it.

```bash
# for the first time execution
cp run_clm.sh.template run_clm.sh

# run the script
bash run_clm.sh
```
