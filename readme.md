# How to run
1. 🐍 Install [conda](https://www.anaconda.com/download) for virtual environment management. Create and activate a new virtual environment.

```shell
conda create -n localGPT python=3.10.0
conda activate localGPT
```

2. 🛠️ Install the dependencies using pip

To set up your environment to run the code, first install all requirements:

```shell
pip install -r requirements.txt
```

3. Set env variable `GOOGLE_API_KEY` to an api key generated in google that will have permissions to the YouTube service

4. Pull all transcripts from Youtube into `./data/transcripts/` folder
```shell
python 1_pull-transcripts.py
```

5. Create embeddings from all transcripts and save to vector storage 
```shell
python 2_create-embeddings.py
```

6. Run the local chatbot!
```shell
python 3_run_llm.py
```

## Running  Mistral 7B with llama.cpp

Tested on Apple Silicon M2 Pro, inspired by [this guide](https://medium.com/@mne/run-mistral-7b-model-on-macbook-m1-pro-with-16gb-ram-using-llama-cpp-44134694b773).

1. Execute steps from 1 to 5 from the base procedure 

2. Download the [quantized GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) version of [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) and save it in _../models/_ directory.

3. Install _cmake_ (use dedicated tool for your OS)
```shell
brew install cmake
```

4. Install _llama-cpp-python_. To get hardware acceleration architectures (e.g. CUDA), refer to _CMAKE_ARGS_ listed in the [official documentation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#installation-with-specific-hardware-acceleration-blas-cuda-metal-etc)
```shell
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

5. Run superfast and accurate local chatbot
```shell
python 3_run-llm-llama-cpp.py
```
or execute single prompt
```shell
python 3_run-llm-llama-cpp.py "Who is Sam Altman?"
```

# Enabling evaluations:
- you need to provide OPENAI_API_KEY as env variable

Docs: https://docs.ragas.io/en/latest/getstarted/evaluation.html
