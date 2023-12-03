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
