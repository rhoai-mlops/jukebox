from locust import HttpUser, task, between
import json

class LoadTestUser(HttpUser):
    @task
    def post_prediction(self):
        json_data = {
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 13],
                    "datatype": "FP32",
                    "data": [[True,274192,0.898,0.472,1,-7.001, 1, 0.0776, 0.0107, 0.0, 0.141, 0.214, 101.061]],
                }
            ]
        }
        self.client.post(
            "/v2/models/jukebox/infer",
            json=json_data,
        )