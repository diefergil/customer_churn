# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project it`s the first project includes in the ML DevOps Engineer Nanodegree from Udacity.
The library tries to build a simple ML pipeline using good practices.
The python dependencies are managed by poetry.
For ensuring good practices, pre-commits hooks was used.
## Running Files

### Locally using poetry

```sh
$ git clone https://github.com/diefergil/customer_churn.git
$ cd /folder
# copy data in /folder/data
$ poetry install
$ poetry run python churn_script_logging_and_tests.py
```

### Using Docker

```sh
$ git clone https://github.com/diefergil/customer_churn.git
$ cd /folder
# copy data in /folder/data
$ docker build --pull --rm -f "Dockerfile" -t customerchurnsolution:latest "."
$ make run_docker
```
