# Murder-Will-Out
Final project for "Models of sequential data" course dedicated to hidden knowledge in LLMs.

## Dependences installation

Our project uses `uv` package manager. You can install it by

```bash
pip install uv
```

Next, to build all dependences you should run the command

```bash
uv sync
```

Finally, to run python scripts, you should do:

```bash
uv run python <script_name>
```

## Data

You can find the original dataset in [EntityQuestions](https://github.com/princeton-nlp/EntityQuestions) repository, or [here](https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip), or using `wget` command:

```bash
wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
```

## Code

### Data preprocessing

Code `data_preprocessing.py` prepares train (500 question), test (500 questions) and dev (500 questions). It filters all dublicates, questions with several answers, and questions where question already has part of golden answer. Also, we checked that train set does not contain questions from test set.

### Generation of greedy answers

Code `greedy_answer_generation.py` generates greedy answers for all questions. 

### Generation of many model's answers

Code `many_answers_generation.py` generates 1000 model's answers for test and dev sets with temperature 1, and generates 500 answers for all train questions with temperature 2.

### LLM judge

Code `label_model_answers.py` labels all answers. First it checks if the answer the exact match with golden answer. If not, answer was labelled using LLM judge.

### Scoring of P(True)

Code `external_scoring.py` calculates scores of P(True) for data.

### Internal scoring

* Code `generate_data_for_probe.py` takes training questions where greedy answers are exact matches of golden one. Next it takes golden answer as positive and several randomly sampled generated answers as negatives. Also it prepares dev dataset. For each Q-A pair code extracts hidden layers and saves them for further linear probe training.
* Code `train_linear_probe.py` trains linear probe, evaluates it on dev set and choose the best layer by the dev score. 
