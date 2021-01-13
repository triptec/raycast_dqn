import os
import multiprocessing

def work(config):
    print(config)
    return os.system(f'cargo run --release -- --stats-file {config["name"]}.csv --actor-layers {config["layer"]} --model-type {config["model-type"]} --max-episodes 10000 --agent-max-age 1000 --render false > /dev/null 2>&1')

configs = [
    {
        "name": "l64a2c",
        "model-type": "a2c",
        "layer": "64",
    },
    {
        "name": "l64-64a2c",
        "model-type": "a2c",
        "layer": "64 64",
    },
    {
        "name": "l128a2c",
        "model-type": "a2c",
        "layer": "128",
    },
    {
        "name": "l128-128a2c",
        "model-type": "a2c",
        "layer": "128 128",
    },
    {
        "name": "l256a2c",
        "model-type": "a2c",
        "layer": "256",
    },
    {
        "name": "l256-256a2c",
        "model-type": "a2c",
        "layer": "256 256",
    },
    {
        "name": "l512a2c",
        "model-type": "a2c",
        "layer": "512",
    },
    {
        "name": "l512-512a2c",
        "model-type": "a2c",
        "layer": "512 512",
    },
    {
        "name": "l1024a2c",
        "model-type": "a2c",
        "layer": "1024",
    },
    {
        "name": "l1024-1024a2c",
        "model-type": "a2c",
        "layer": "1024 1024",
    },
    {
        "name": "l64ddqn",
        "model-type": "ddqn",
        "layer": "64",
    },
    {
        "name": "l64-64ddqn",
        "model-type": "ddqn",
        "layer": "64 64",
    },
    {
        "name": "l128ddqn",
        "model-type": "ddqn",
        "layer": "128",
    },
    {
        "name": "l128-128ddqn",
        "model-type": "ddqn",
        "layer": "128 128",
    },
    {
        "name": "l256ddqn",
        "model-type": "ddqn",
        "layer": "256",
    },
    {
        "name": "l256-256ddqn",
        "model-type": "ddqn",
        "layer": "256 256",
    },
    {
        "name": "l512ddqn",
        "model-type": "ddqn",
        "layer": "512",
    },
    {
        "name": "l512-512ddqn",
        "model-type": "ddqn",
        "layer": "512 512",
    },
    {
        "name": "l1024ddqn",
        "model-type": "ddqn",
        "layer": "1024",
    },
    {
        "name": "l1024-1024ddqn",
        "model-type": "ddqn",
        "layer": "1024 1024",
    },
]

with multiprocessing.Pool(4) as pool:
    pool.map(work, configs)
