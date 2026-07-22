import pytest
from flask_app import create_app
import json
import logging

logging.basicConfig(level=logging.INFO)

def test_scan():
    app = create_app(testing=True)
    with app.test_client() as client:
        print("Starting scan...")
        resp = client.post("/api/scan", json={"sample": "asset3.mp4"})
        print(f"Status Code: {resp.status_code}")
        try:
            print(f"Response: {resp.json}")
        except:
            print(f"Response Text: {resp.text}")

if __name__ == "__main__":
    test_scan()
