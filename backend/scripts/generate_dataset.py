import json
import random

BUG_TYPES = [
    "logic",
    "performance",
    "security",
    "concurrency",
    "memory",
    "style",
    "none"
]

def rand_var():
    return random.choice(["x", "y", "data", "value", "count", "temp", "num"])

def logic_bug():
    a = rand_var()
    b = rand_var()
    return {
        "code": f"def divide({a}, {b}):\n    return {a} / {b}\n\nprint(divide(10, 0))",
        "language": "python",
        "bugs": ["logic"],
        "severity": round(random.uniform(0.75, 0.95), 2),
        "complexity_before": "O(1)",
        "complexity_after": "O(1)"
    }

def performance_bug():
    v = rand_var()
    return {
        "code": f"for i in range(n):\n    for j in range(n):\n        print(i, j)",
        "language": "python",
        "bugs": ["performance"],
        "severity": round(random.uniform(0.6, 0.85), 2),
        "complexity_before": "O(n^2)",
        "complexity_after": "O(n)"
    }

def security_bug():
    return {
        "code": "user_input = input()\nquery = \"SELECT * FROM users WHERE id=\" + user_input\nexecute(query)",
        "language": "python",
        "bugs": ["security"],
        "severity": round(random.uniform(0.8, 0.98), 2),
        "complexity_before": "O(1)",
        "complexity_after": "O(1)"
    }

def concurrency_bug():
    return {
        "code": "counter = 0\ndef increment():\n    global counter\n    counter += 1",
        "language": "python",
        "bugs": ["concurrency"],
        "severity": round(random.uniform(0.65, 0.9), 2),
        "complexity_before": "O(1)",
        "complexity_after": "O(1)"
    }

def memory_bug():
    return {
        "code": "def create_list():\n    data = []\n    while True:\n        data.append('leak')",
        "language": "python",
        "bugs": ["memory"],
        "severity": round(random.uniform(0.7, 0.9), 2),
        "complexity_before": "O(n)",
        "complexity_after": "O(n)"
    }

def style_bug():
    return {
        "code": "def Foo():\n x=1\n return(x)",
        "language": "python",
        "bugs": ["style"],
        "severity": round(random.uniform(0.1, 0.4), 2),
        "complexity_before": "O(1)",
        "complexity_after": "O(1)"
    }

def clean_sample():
    return {
        "code": "def add(a, b):\n    return a + b\n\nprint(add(5, 3))",
        "language": "python",
        "bugs": ["none"],
        "severity": round(random.uniform(0.0, 0.1), 2),
        "complexity_before": "O(1)",
        "complexity_after": "O(1)"
    }

GENERATORS = [
    logic_bug,
    performance_bug,
    security_bug,
    concurrency_bug,
    memory_bug,
    style_bug,
    clean_sample
]

def generate_dataset(size=500):
    samples = []
    for _ in range(size):
        generator = random.choice(GENERATORS)
        samples.append(generator())
    return samples

if __name__ == "__main__":
    dataset = generate_dataset(500)

    train = dataset[:400]
    val = dataset[400:450]
    test = dataset[450:]

    with open("data/processed/train.jsonl", "w") as f:
        for sample in train:
            f.write(json.dumps(sample) + "\n")

    with open("data/processed/val.jsonl", "w") as f:
        for sample in val:
            f.write(json.dumps(sample) + "\n")

    with open("data/processed/test.jsonl", "w") as f:
        for sample in test:
            f.write(json.dumps(sample) + "\n")

    print("✅ Generated 500-sample dataset (400 train / 50 val / 50 test)")
