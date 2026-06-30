import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
ARENA_RUNS_DIR = os.path.join(DATA_DIR, "arena_runs")

class Database:
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(RUNS_DIR, exist_ok=True)
        os.makedirs(ARENA_RUNS_DIR, exist_ok=True)
        self.initialize_default_datasets()

    def _safe_json_path(self, base_dir: str, resource_id: str) -> Optional[str]:
        filename = f"{resource_id}.json"
        base_abs = os.path.abspath(base_dir)
        candidate_abs = os.path.abspath(os.path.join(base_abs, filename))
        if os.path.commonpath([base_abs, candidate_abs]) != base_abs:
            return None
        return candidate_abs

    def initialize_default_datasets(self):
        """Seed the system with some high-quality datasets if empty/missing."""
        default_files = os.listdir(DATASETS_DIR)
        
        # 1. Logical Reasoning & Puzzles Benchmark
        has_logic = any(f.endswith(".json") and "logical_reasoning" in f for f in default_files)
        if not has_logic:
            logical_dataset = {
                "id": "logical_reasoning_benchmark",
                "name": "Logical Reasoning & Puzzles Benchmark",
                "description": "A curated dataset of high-quality logical puzzles, mathematical reasoning, and algorithmic logic questions.",
                "created_at": datetime.utcnow().isoformat(),
                "cases": [
                    {
                        "id": "logic-01",
                        "question": "A box contains 3 red balls and 7 blue balls. A player randomly draws two balls from the box one after another without replacement. What is the probability that both balls drawn are red? Express your answer as a simplified fraction.",
                        "ideal_answer": "The probability is 1/15. \n\nProof:\n1. The probability of drawing a red ball on the first draw is 3/10 (3 red balls out of 10 total balls).\n2. Since the drawing is without replacement, there are now 2 red balls and 7 blue balls left, making a total of 9 balls.\n3. The probability of drawing a red ball on the second draw is 2/9.\n4. The joint probability of drawing two red balls is: (3/10) * (2/9) = 6/90.\n5. Simplifying 6/90 by dividing the numerator and denominator by 6 yields 1/15."
                    },
                    {
                        "id": "logic-02",
                        "question": "You stand in front of two doors. One door leads to heaven, and the other leads to hell. In front of each door is a guard. One guard always tells the truth, and the other guard always lies. You do not know which guard is which, or which door leads where. You are allowed to ask exactly one guard exactly one question to find the door to heaven. What question should you ask?",
                        "ideal_answer": "You should point to one of the doors and ask either guard: 'If I were to ask the other guard if this door leads to heaven, what would they say?'\n\nExplanation:\n1. If the door you pointed to indeed leads to heaven:\n   - The truth-teller would know the liar would say 'No'. Thus, the truth-teller answers 'No'.\n   - The liar would know the truth-teller would say 'Yes'. Since the liar must lie, they answer 'No'.\n2. If the door you pointed to leads to hell:\n   - The truth-teller would know the liar would say 'Yes'. Thus, the truth-teller answers 'Yes'.\n   - The liar would know the truth-teller would say 'No'. Since the liar must lie, they answer 'Yes'.\n\nIn both cases, both guards will give the exact same answer: 'No' if the door leads to heaven, and 'Yes' if the door leads to hell. Therefore, you should choose the door you pointed to if they answer 'No', and the other door if they answer 'Yes'."
                    },
                    {
                        "id": "logic-03",
                        "question": "A farmer needs to cross a river with a wolf, a goat, and a box of cabbage. His boat is small and can only hold himself and one of the three items at a time. If left unattended, the wolf will eat the goat, and the goat will eat the cabbage. How can the farmer get all three items safely to the other side of the river? Outline the step-by-step trips.",
                        "ideal_answer": "Here is the step-by-step solution to safely cross the river:\n\n1. Take the goat across: The farmer takes the goat to the other side, leaving the wolf and cabbage together (safe). The farmer returns alone.\n2. Take the wolf across and bring back the goat: The farmer takes the wolf to the other side, leaves the wolf, and takes the goat back to the starting side (preventing the wolf from eating the goat). \n3. Take the cabbage across: The farmer leaves the goat at the start, takes the cabbage to the other side, leaving it with the wolf (safe). The farmer returns alone.\n4. Take the goat across: The farmer takes the goat across to the other side for the final time. All three items are now safely on the other side."
                    },
                    {
                        "id": "logic-04",
                        "question": "An algorithmic function receives a positive integer n. If n is even, it divides it by 2. If n is odd, it multiplies it by 3 and adds 1. This process is repeated. Write a Python function `collatz_steps(n)` that returns the number of steps required to reach the number 1. If n is 1, it should return 0.",
                        "ideal_answer": "Here is the Python implementation using a simple loop:\n\n```python\ndef collatz_steps(n: int) -> int:\n    if n <= 0:\n        raise ValueError(\"n must be a positive integer\")\n    steps = 0\n    while n > 1:\n        if n % 2 == 0:\n            n = n // 2\n        else:\n            n = n * 3 + 1\n        steps += 1\n    return steps\n```\n\nExplanation:\n- We initialize a step counter to 0.\n- A `while` loop runs as long as `n` is greater than 1.\n- In each iteration, we apply the Collatz sequence rule (divide by 2 if even, or triple plus 1 if odd) and increment the step counter.\n- The loop terminates when `n` becomes 1, returning the total steps."
                    },
                    {
                        "id": "logic-05",
                        "question": "Four people need to cross a suspension bridge at night. They have only one flashlight, and the bridge is only strong enough to support two people at a time. Any crossing must be done with the flashlight. The four people walk at different speeds: Alice takes 1 minute to cross, Bob takes 2 minutes, Charlie takes 5 minutes, and Daniel takes 10 minutes. When two people cross together, they must walk at the slower person's pace. What is the minimum time (in minutes) required for all four to cross the bridge?",
                        "ideal_answer": "The minimum time required is 17 minutes.\n\nHere is the optimal sequence of crossings:\n1. Alice and Bob cross the bridge with the flashlight (takes 2 minutes). Alice and Bob are on the other side.\n2. Alice returns with the flashlight (takes 1 minute). Alice is back; Bob is on the other side (Total time: 3 mins).\n3. Charlie and Daniel cross the bridge with the flashlight (takes 10 minutes). Charlie, Daniel, and Bob are on the other side.\n4. Bob returns with the flashlight (takes 2 minutes). Bob is back; Charlie and Daniel are on the other side (Total time: 15 mins).\n5. Alice and Bob cross the bridge with the flashlight (takes 2 minutes). All four are on the other side.\n\nTotal time: 2 + 1 + 10 + 2 + 2 = 17 minutes."
                    }
                ]
            }
            self.save_dataset(logical_dataset)

        # 2. Multi-Turn Conversational Benchmark
        has_multiturn = any(f.endswith(".json") and "multi_turn" in f for f in default_files)
        if not has_multiturn:
            multiturn_dataset = {
                "id": "multi_turn_conversational_benchmark",
                "name": "Multi-Turn Conversational Benchmark",
                "description": "A curated dataset for evaluating multi-turn chat capabilities, context retention, and instruction following across multiple turns.",
                "created_at": datetime.utcnow().isoformat(),
                "cases": [
                    {
                        "id": "conv-01",
                        "turns": [
                            {
                                "user_message": "I am planning a 3-day trip to Paris. Can you suggest a brief daily itinerary?",
                                "ideal_response": "Here is a brief 3-day Paris itinerary:\n- Day 1: Visit the Eiffel Tower in the morning, walk along the Seine River, and explore the Louvre Museum in the afternoon.\n- Day 2: Visit Notre-Dame Cathedral, stroll through the Latin Quarter, and enjoy a scenic evening Seine river cruise.\n- Day 3: Explore the artistic streets of Montmartre, visit the Sacré-Cœur Basilica, and spend the afternoon at the Palace of Versailles."
                            },
                            {
                                "user_message": "That sounds great! For Day 1, can you recommend a good lunch spot near the Louvre?",
                                "ideal_response": "Certainly! Near the Louvre, you can try:\n1. Café Marly: Located right under the arcades of the Louvre, offering stunning views of the glass pyramid.\n2. Bistrot Victoires: A short walk away, famous for its classic French bistro atmosphere and affordable, delicious steak frites."
                            },
                            {
                                "user_message": "How far is the second restaurant from the Louvre by foot?",
                                "ideal_response": "Bistrot Victoires is located at 6 Rue de la Vrillière. It is approximately a 7 to 10-minute walk (about 600 meters) from the Louvre Museum's main pyramid entrance."
                            }
                        ]
                    },
                    {
                        "id": "conv-02",
                        "turns": [
                            {
                                "user_message": "I have a Python list of numbers: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`. I want to filter out the odd numbers and keep only the even ones. How do I do this using a list comprehension?",
                                "ideal_response": "You can filter the even numbers using this list comprehension:\n```python\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nevens = [x for x in numbers if x % 2 == 0]\nprint(evens)  # Output: [2, 4, 6, 8, 10]\n```"
                            },
                            {
                                "user_message": "Can you rewrite that using the `filter()` function instead?",
                                "ideal_response": "Yes! Here is the same operation using the `filter()` function combined with a `lambda` expression:\n```python\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nevens = list(filter(lambda x: x % 2 == 0, numbers))\nprint(evens)  # Output: [2, 4, 6, 8, 10]\n```"
                            },
                            {
                                "user_message": "Which of these two approaches is generally considered more Pythonic and faster?",
                                "ideal_response": "The list comprehension `[x for x in numbers if x % 2 == 0]` is generally considered more Pythonic and is faster.\n\nWhy it's faster:\n- List comprehensions run at near-C speed inside the interpreter.\n- `filter` with a `lambda` requires a Python function call for every single item in the list, which adds substantial overhead.\n\nWhy it's more Pythonic:\n- List comprehensions are highly readable and the preferred way to filter/map in modern Python."
                            }
                        ]
                    }
                ]
            }
            self.save_dataset(multiturn_dataset)

    # Dataset Methods
    def get_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        for filename in os.listdir(DATASETS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(DATASETS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        datasets.append(json.load(f))
                except Exception as e:
                    print(f"Error loading dataset {filename}: {e}")
        return sorted(datasets, key=lambda x: x.get("name", ""))

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        filepath = os.path.join(DATASETS_DIR, f"{dataset_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading dataset {dataset_id}: {e}")
        return None

    def save_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in dataset or not dataset["id"]:
            dataset["id"] = str(uuid.uuid4())
        if "created_at" not in dataset:
            dataset["created_at"] = datetime.utcnow().isoformat()
        
        filepath = os.path.join(DATASETS_DIR, f"{dataset['id']}.json")
        with open(filepath, "w") as f:
            json.dump(dataset, f, indent=2)
        return dataset

    def delete_dataset(self, dataset_id: str) -> bool:
        filepath = self._safe_json_path(DATASETS_DIR, dataset_id)
        if not filepath:
            return False
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    # Evaluation Run Methods
    def get_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for filename in os.listdir(RUNS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(RUNS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        summary = {k: v for k, v in data.items() if k != "results"}
                        runs.append(summary)
                except Exception as e:
                    print(f"Error loading run {filename}: {e}")
        return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        filepath = self._safe_json_path(RUNS_DIR, run_id)
        if not filepath:
            return None
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading run {run_id}: {e}")
        return None

    def save_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in run or not run["id"]:
            run["id"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        if "created_at" not in run:
            run["created_at"] = datetime.utcnow().isoformat()

        filepath = self._safe_json_path(RUNS_DIR, run["id"])
        if not filepath:
            raise ValueError("Invalid run id")
        with open(filepath, "w") as f:
            json.dump(run, f, indent=2)
        return run

    def delete_run(self, run_id: str) -> bool:
        filepath = self._safe_json_path(RUNS_DIR, run_id)
        if not filepath:
            return False
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    # Arena Run Methods
    def get_arena_runs(self) -> List[Dict[str, Any]]:
        """Returns summary list of arena runs (results array excluded for performance)."""
        runs = []
        for filename in os.listdir(ARENA_RUNS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(ARENA_RUNS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        summary = {k: v for k, v in data.items() if k != "results"}
                        runs.append(summary)
                except Exception as e:
                    print(f"Error loading arena run {filename}: {e}")
        return sorted(runs, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_arena_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        filepath = self._safe_json_path(ARENA_RUNS_DIR, run_id)
        if not filepath:
            return None
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading arena run {run_id}: {e}")
        return None

    def save_arena_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in run or not run["id"]:
            run["id"] = f"arena_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        if "created_at" not in run:
            run["created_at"] = datetime.utcnow().isoformat()

        filepath = self._safe_json_path(ARENA_RUNS_DIR, run["id"])
        if not filepath:
            raise ValueError("Invalid arena run id")
        with open(filepath, "w") as f:
            json.dump(run, f, indent=2)
        return run

    def delete_arena_run(self, run_id: str) -> bool:
        filepath = self._safe_json_path(ARENA_RUNS_DIR, run_id)
        if not filepath:
            return False
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
