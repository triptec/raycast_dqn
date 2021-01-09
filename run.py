import os

layers = ["128", "256", "512", "128 128", "256 256", "1024"]

for layer in layers:
    print(layer)
    os.system(f'cargo run --release -- --stats-file {layer.replace(" ","-")}.csv --actor-layers {layer} --critic-layers {layer} --max-episodes 20000')
