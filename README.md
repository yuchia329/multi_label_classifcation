# Multi-label Classification

The project is multi-label classification task for NLP 243 – Deep Learning for Natural Language Processing. The objective for the project is finding the relation between hyperparameters and performance.

## Description

The training dataset contains 2300+ utterances and core_relations. The testing dataset contains 980+ utterances.\
The model used for the training is self-defined MLP model. The evaluation is on Kaggle. The best model by far in the experiment is in run.py, but there are room for higher accuracy with MLP model.

## Installation

Create virtual environment with Conda or Venv. For MacOS with Intel processor, run

```
conda create --platform osx-64 --name NLP243_project1 python=3.12.1
```

Install packages

```
pip install -r src/requirement.txt
```

execute run.py with paths to "train_file", "test_file", and "output_file"

```
python src/run.py hw1_train-1.csv hw1_test-2.csv predict/output.csv
```

predict folder contains working files during the training.
