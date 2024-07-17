import paramiko
import os
import requests
from time import sleep
deployment_url = os.environ["airflow_url"]
print(deployment_url)
print(os.environ['airflow_username'])
print(os.environ['airflow_password'])
dag_id = "term_deposit_prediction_JyothsnaPendyala"
# sleep(5)
response = requests.post(
    url=f"{deployment_url}/api/v1/dags/{dag_id}/dagRuns",
    auth=(os.environ['airflow_username'], os.environ['airflow_password']),
    verify=False,
    json={
        "conf": {},
    }
)
print(response)
