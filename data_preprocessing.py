import json
from pathlib import Path
import os
import random
from typing import Dict, Tuple, Union, List
from collections import defaultdict as ddict
from tqdm.auto import tqdm

CATEGS = ["P176", "P264", "P50", "P40"]

INITIAL_DATA_PATH = "./data/dataset"
FINAL_PATH = "./data/final_dataset"

MODES = ['train', 'test']
TRAIN_SAMPLE = 2000
TEST_SAMPLE = 500
DEV_SAMPLE = 50

UNIQUE_QUESTIONS_TEST = ddict(list)
UNIQUE_TRIPLETS_TEST = ddict(list)

TRIPLETS_DEV = ddict(list)

UNIQUE_QUESTIONS_TRAIN = ddict(list)
UNIQUE_TRIPLETS_TRAIN = ddict(list)

COMMON = [
    "A", "An", "The", "Is", "Of", "For"
]


def filter_by_answer(entry:Dict) -> bool:
    if (len(entry["answers"]) > 1) or (len(entry["answers"]) == 0):
        return False
    answer = entry["answers"][0]
    question = entry["question"]
    if answer.lower() in question.lower():
        return False
    for word in answer.split(" "):
        if word.istitle() and (len(word) > 2) and (word in question.split(" ")) and (word not in COMMON):
            print(word)
            return False
    return True

def check_marriage(entry:Dict) -> bool:
    for elem in UNIQUE_TRIPLETS_TEST["P26"]:
        if elem["answers"][0].lower() in entry["question"].lower():
            return True
        if entry["answers"][0].lower() in elem["question"].lower():
            return True
    return False


def load_json(path:Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def makedirs() -> None:
    for path in (FINAL_PATH,):
        for mode in ("train", "test", "dev"):
            curr_path = Path(path) / mode
            os.makedirs(curr_path, exist_ok=True)


def filter_test() -> None:
    test_path = Path(INITIAL_DATA_PATH) / "test"
    for file in tqdm(os.listdir(test_path)):
        categ = file[:file.find(".")]
        if categ in CATEGS:
            data = load_json(test_path / file)
            for entry in data:
                if filter_by_answer(entry):
                    if entry["question"].lower() not in UNIQUE_QUESTIONS_TEST[categ]:
                        UNIQUE_QUESTIONS_TEST[categ].append(entry["question"].lower())
                        UNIQUE_TRIPLETS_TEST[categ].append(entry)

def filter_train() -> None:
    train_path = Path(INITIAL_DATA_PATH) / "train"
    for file in tqdm(os.listdir(train_path)):
        categ = file[:file.find(".")]
        if categ in CATEGS:
            data = load_json(train_path / file)
            for entry in data:
                if filter_by_answer(entry):
                    if entry["question"].lower() not in UNIQUE_QUESTIONS_TEST[categ]:
                        if entry["question"].lower() not in UNIQUE_QUESTIONS_TRAIN[categ]:
                            if (categ == "P26") and check_marriage(entry):
                                continue
                            UNIQUE_QUESTIONS_TRAIN[categ].append(entry["question"].lower())
                            UNIQUE_TRIPLETS_TRAIN[categ].append(entry)

def final_sampling() -> Tuple[ddict]:
    """Sample data for train, test, and dev splits from filtered datasets.
    
    Returns:
        Tuple containing final_test, final_dev, and final_train datasets
    """
    final_test = ddict(list)
    final_dev = ddict(list)
    final_train = ddict(list)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    for categ in CATEGS:
        # Sample from test data for test and dev splits
        test_data = UNIQUE_TRIPLETS_TEST[categ]
        if len(test_data) >= TEST_SAMPLE + DEV_SAMPLE:
            # Randomly sample test data
            sampled_test = random.sample(test_data, TEST_SAMPLE)
            final_test[categ] = sampled_test
            
            # Sample dev data from remaining test data
            remaining_test = [item for item in test_data if item not in sampled_test]
            if len(remaining_test) >= DEV_SAMPLE:
                sampled_dev = random.sample(remaining_test, DEV_SAMPLE)
                final_dev[categ] = sampled_dev
            else:
                final_dev[categ] = remaining_test
        else:
            # If not enough test data, still respect TEST_SAMPLE limit
            if len(test_data) >= TEST_SAMPLE:
                sampled_test = random.sample(test_data, TEST_SAMPLE)
                final_test[categ] = sampled_test
                # Use remaining data for dev if any
                remaining_test = [item for item in test_data if item not in sampled_test]
                final_dev[categ] = remaining_test
            else:
                # Use all available data for test, none for dev
                final_test[categ] = test_data
                final_dev[categ] = []
        
        # Sample from train data
        train_data = UNIQUE_TRIPLETS_TRAIN[categ]
        if len(train_data) >= TRAIN_SAMPLE:
            sampled_train = random.sample(train_data, TRAIN_SAMPLE)
            final_train[categ] = sampled_train
        else:
            final_train[categ] = train_data
    
    return final_test, final_dev, final_train


def save_data(final_test: ddict, final_dev: ddict, final_train: ddict) -> None:
    """Save final data splits as JSON files in the specified directory structure.
    
    Args:
        final_test: Dictionary containing test data for each category
        final_dev: Dictionary containing dev data for each category  
        final_train: Dictionary containing train data for each category
    """
    for mode, data_dict in [("test", final_test), ("dev", final_dev), ("train", final_train)]:
        mode_path = Path(FINAL_PATH) / mode
        os.makedirs(mode_path, exist_ok=True)
        
        for categ in CATEGS:
            if data_dict[categ]:  # Only save if there's data for this category
                output_file = mode_path / f"{categ}.{mode}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data_dict[categ], f, indent=2, ensure_ascii=False)
                print(f"Saved {len(data_dict[categ])} entries to {output_file}")



def main() -> None:
    """Main function to orchestrate the data preprocessing pipeline."""
    print("Starting data preprocessing...")
    
    # Create output directories
    makedirs()
    
    # Filter and collect unique data from train and test sets
    print("Filtering train data...")
    filter_train()
    print("Filtering test data...")
    filter_test()
    
    # Sample data for final splits
    print("Sampling data for train/test/dev splits...")
    final_test, final_dev, final_train = final_sampling()
    
    # Save the final datasets
    print("Saving final datasets...")
    save_data(final_test, final_dev, final_train)
    
    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
