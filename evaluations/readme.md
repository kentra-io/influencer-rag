# Evaluations
We're using Ragas for evaluation.
If you'd like to know more about RAG evaluations, please read the [Ragas documentation](https://docs.ragas.io/en/latest/getstarted/evaluation.html).

## Enable evaluations when running the application
1. install the library
```shell
pip install ragas
```

2. Provide `OPENAI_API_KEY` as env variable
3. Enable evaluation in _config.py_ by setting `evaluations_enabled` to `True`

The result will be that:
- evaluations will be done for every question you ask. 
- the results will be stored in data/evaluations/{channel_handle}/questions.csv

TODO: implement handling of channels other than BenFelixCSI

## Run evaluations on all questions
Questions you have asked are stored in data/questions/{channel_handle}/questions.csv. You can enrich them with ground truths, but even without that you can evaluate the quality of current setup of the application. You can also add new questions to that file.
To do that, in the root folder of the project run:
```shell
python evaluate_all_questions.py
```

###TODO: 
Implement configuration labels. The RagConfiguration class should allow you to label the retrieval and the llm part so that you can have evaluations for different configurations in separate files.

###TODO: 
Implement utilizing ground truths and metrics relying on them
