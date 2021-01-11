import os
import multiprocessing

def work(config):
    print(config)
    return os.system(f'cargo run --release -- --stats-file {config["name"]}.csv --actor-layers {config["layer"]} --critic-layers {config["layer"]} --gamma {config["gamma"]} --max-episodes 30000 --render false > /dev/null 2>&1')

configs = [
    {
        "name": "l64-64-64-g999",
        "layer": "64 64 64",
        "gamma": "0.999",
     },
#    {
#        "name": "l64-64-g999",
#        "layer": "64 64",
#        "gamma": "0.999",
#    },
#    {
#        "name": "l256-g999",
#        "layer": "256",
#        "gamma": "0.999",
#    },
#    {
#        "name": "l256-256-g999",
#        "layer": "256 256",
#        "gamma": "0.999",
#    },
#    {
#        "name": "l128-g999",
#        "layer": "128",
#        "gamma": "0.999",
#    },
#    {
#        "name": "l128-128-g999",
#        "layer": "128 128",
#        "gamma": "0.999",
#    },
]

with multiprocessing.Pool(2) as pool:
    pool.map(work, configs)
