# How to run
## Prerequisites
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

## Running chatbot on Mistral 7B with llama.cpp

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

5. Run local Mistral 7B chatbot
```shell
 python _3_run_llm_llama_cpp.py
 ```
or execute single prompt
```shell
 python _3_run_llm_llama_cpp.py "Who is Sam Altman?"
```

The code was written for Mistral 7B, but any other GGUF model should also work. Just download the model to `../models/` and update `model_name` in `config.py`.

## Other LLM providers
The default configuration runs llama-cpp, however there are 2 more implementations allowing to run LLMs from different vendors.

### OpenAI
Running LLM from OpenAI API is the first option. Please follow the steps to run:

1. Update the `model_name` in `config.py` to one of OpenAI models (currently, only _gpt-3.5-turbo_ was tested)
2. Add `OPENAI_API_KEY` variable with your API Key
3. Make sure `openai` is installed (should be already there as one of the existing transitive dependencies)
```shell
pip install openai
```

5. Run OpenAI chatbot
```shell
python _3_run-llm-llama-cpp.py
```

### HF Transformers
1. Update the `model_name` in `config.py` to one of HF Transformers models (e.g.  _ericzzz/falcon-rw-1b-instruct-openorca_)

2. Run local chatbot with HF Transformers
```shell
python 3_run-llm-llama-cpp.py
```

## Enabling evaluations
Evaluations have a separate [readme](evaluations/readme.md)

## Using Streamlit UI

1. Install streamlit
```shell
pip install streamlit
```

2. Run streamlit app
```shell
streamlit run ui.py
```

## Using ElasticSearch instead of ChromaDB
1. Run:
```shell
docker compose up -d
```
to start ElasticSearch and Kibana

See [elasticsearch.md](/docs/elasticsearch) for example queries.

2. Modify `config.py` - `default_vector_db` variable to make ELASTICSEARCH a default vector store.
