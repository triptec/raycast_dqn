import os
import multiprocessing


def work(config):
    print(config)
    return os.system(
        f'cargo run --release -- --stats-file {config["model-type"]}-{config["layer"].replace(" ", "-")}-pm{config["prioritized-memory"]}.csv --prioritized-memory {config["prioritized-memory"]} --actor-layers {config["layer"]} --model-type {config["model-type"]} --max-episodes 10000 --agent-max-age 1000 --render false > /dev/null 2>&1')


layers = ["512 512", "1024 1024"]
models = ["a2c", "ddqn"]
prioritized_memory = ["true", "false"]

configs = []

for layer in layers:
    for model in models:
        for pm in prioritized_memory:
            configs.push({
                "layer": layer,
                "model-type": model,
                "prioritized-memory": pm,
            })

with multiprocessing.Pool(4) as pool:
    pool.map(work, configs)
