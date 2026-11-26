"""Simulate many users sending mosquito recordings to the model.

Each demo recording's true class is in its file name (demo_<class>.wav), and a
response only counts as a success when the predicted class is correct.

Example: locust -f load_tests/locustfile.py --host http://127.0.0.1:8080
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = [
    (path.name, path.stem.replace("demo_", ""), path.read_bytes())
    for path in sorted((ROOT / "demo" / "predict").glob("demo_*.wav"))
]


class FieldUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, label, content = random.choice(RECORDINGS)
        with self.client.post(
            "/predict",
            files={"file": (name, content, "audio/wav")},
            name="POST /predict",
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif response.json().get("label") != label:
                response.failure("Wrong label")
            else:
                response.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health", timeout=60)
