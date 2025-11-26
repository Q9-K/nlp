import wandb
import os

run = wandb.init(project="nlp_hw2", name="test_run")
run.log({"example_metric": 0.5})
run.finish()