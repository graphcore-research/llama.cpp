import matplotlib.pyplot as plt
import numpy as np


files = ["./cpu-dense-attn-benchmarks-implementation-fast-extender.txt", "./cpu-dense-attn-benchmarks-implementation-slow-iterative.txt"]

for filename in files:
    results = {}
    with open(filename, "r") as file:
        for line in file.readlines():
            seconds = float(line.split(": ")[-1])
            prompt = int(line.split(": ")[0].split(" ")[-1])
            results[prompt] = results.get(prompt, []) + [seconds]

    truncated_results = {}
    for key, val in results.items():
        truncated_results[key] = sorted(val)[50:-50]
        print(key, len(truncated_results[key]))

    x = truncated_results.keys()
    y = [np.mean(truncated_results[i]) for i in x]
    lower = [np.mean(truncated_results[i]) - np.std(truncated_results[i]) for i in x]
    upper = [np.mean(truncated_results[i]) + np.std(truncated_results[i]) for i in x]
    plt.fill_between(x, lower, upper, alpha=0.4)
    plt.plot(x, y, label=" ".join(filename.split("-")[-2:])[:-4])
plt.legend()
plt.savefig("./comparison.png", dpi=400)