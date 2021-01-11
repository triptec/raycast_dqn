import os
import multiprocessing

def work(config):
    print(config)
    return os.system(f'cargo run --release -- --stats-file {config["name"]}.csv --actor-layers {config["layer"]} --max-episodes 10000 --agent-max-age --render false > /dev/null 2>&1')

configs = [
    {
        "name": "l64",
        "layer": "64",
    },
    {
        "name": "l64-64",
        "layer": "64 64",
    },
    {
        "name": "l128",
        "layer": "128",
    },
    {
        "name": "l128-128",
        "layer": "128 128",
    },
    {
        "name": "l256",
        "layer": "256",
    },
    {
        "name": "l256-256",
        "layer": "256 256",
    },
    {
        "name": "l512",
        "layer": "512",
    },
    {
        "name": "l512-512",
        "layer": "512 512",
    },
    {
        "name": "l1024",
        "layer": "1024",
    },
    {
        "name": "l1024-1024",
        "layer": "1024 1024",
    },
]

with multiprocessing.Pool(2) as pool:
    pool.map(work, configs)
