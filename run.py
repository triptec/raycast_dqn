import os
import multiprocessing


def work(config):
    print(config)
    return os.system(
        f'cargo run --release -- --stats-file {config["model-type"]}-{config["layer"].replace(" ", "-")}.csv --actor-layers {config["layer"]} --model-type {config["model-type"]} --max-episodes 10000 --agent-max-age 1000 --render false > /dev/null 2>&1')


layers = ["64", "64 64", "512", "512 512", "1024", "1024 1024"]
models = ["a2c", "ddqn"]

configs = []

for layer in layers:
    for model in models:
        configs.push({
            "layer": layer,
            "model-type": model,
        })

with multiprocessing.Pool(4) as pool:
    pool.map(work, configs)
