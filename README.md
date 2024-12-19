# Huggingface Transformers Starter Code

General starter code for creative model architecture with huggingface transformer library. Users may implement customized llama model in `models` directory.

## Installation

Change the cuda version if it is not compatible. Developped with python 3.12.4.

```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
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
