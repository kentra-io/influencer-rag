# Configuration
If you're using `conda`, you need to configure conda and pytest. 
```shell
conda install -c anaconda pytest
```
you can verify if it's set up correctly by running
```shell
which pytest
```
By default, console output is enabled in tests. To disable it, set `log_cli = false` in pytest.ini

# Running tests
To run tests, do
```shell
pytest tests
```
To run a single test, do
```shell
pytest tests/evaluate_all_questions_test.py
```

# Writing tests
In your test file, all methods starting with test_ will be run as tests. 

# Project setup
In the root dir of the project, the file pytest.ini contains the configuration for pytest. Currently only logger configuration.

In the tests directory file confitest.py  
