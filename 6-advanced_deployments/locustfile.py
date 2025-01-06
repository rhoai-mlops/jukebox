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
                    "data": [1.0, 0.27927992323571016, 0.9089068825910931, 0.472935276552163, 0.09090909090909091,0.7477259841743289, 1.0, 0.08425624321389794, 0.010735492809498851, 0.0, 0.13183279742765275, 0.21638018200202225, 0.42839339231137696],
                }
            ]
        }
        self.client.post(
            "/v2/models/jukebox/infer",
            json=json_data,
        )