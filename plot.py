import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import ijson
from tqdm import tqdm


MAX_BAD_QUERIES = 1000
MAX_SAMPLES = 100

def process(dists, bad):
    dists[bad] = np.inf
    dists = np.minimum.accumulate(dists)
    return dists

def clamp(arr, size):
    if len(arr) >= size:
        return arr[:size]
    else:
        new_arr = np.zeros(size, dtype=arr.dtype)
        new_arr[:len(arr)] = arr
        new_arr[len(arr):] = arr[-1]
        return new_arr

def is_safe(ddd):
    s = ddd['safe']
    if isinstance(s, list):
        return s[0]
    else:
        return s

def keep_if_simulated(ddd):
    if is_safe(ddd):
        return True
    return ddd["equivalent_simulated_queries"] == 1

def load(dir):
    with open(f"{dir}/distances_traces.json", "rb") as f:
        samples = ijson.items(f, "item")

        all_dists_bad = []
        all_dists_bad_sim = []

        for sample in tqdm(samples, total=MAX_SAMPLES):
            dists = np.asarray([q['distance'] for q in sample])
            bad = np.asarray([not is_safe(q) for q in sample])
            best_dists = process(dists, bad)
            dists_bad = clamp(best_dists[bad], MAX_BAD_QUERIES)
            all_dists_bad.append(dists_bad)

            if 'equivalent_simulated_queries' in sample[0]:
                dists_sim = np.asarray([q['distance'] for q in sample if keep_if_simulated(q)])
                bad_sim = np.asarray([not is_safe(q) for q in sample if keep_if_simulated(q)])
                best_dists_sim = process(dists_sim, bad_sim)
                dists_bad_sim = clamp(best_dists_sim[bad_sim], MAX_BAD_QUERIES)
                all_dists_bad_sim.append(dists_bad_sim)

            if len(all_dists_bad) >= MAX_SAMPLES:
                break

        return np.asarray(all_dists_bad), np.asarray(all_dists_bad_sim)

"""
def load(dir, simulate=False, d=None):
    if d is None:
        d = json.load(open(f"{dir}/distances_traces.json"))[:MAX_SAMPLES]
    #print(d[0][0:10])

    # DIMS = NUM_IMAGES x NUM_QUERIES
    best_dists = [np.asarray([ddd['distance'] for ddd in dd if not simulate or keep_if_simulated(ddd)]) for dd in d]

    # DIMS = NUM_IMAGES x NUM_QUERIES
    bad = [np.asarray([not is_safe(ddd) for ddd in dd if not simulate or keep_if_simulated(ddd)]) for dd in d]

    best_dists = [process(best_dists[i], bad[i]) for i in range(len(best_dists))]

    #print([sum(b) for b in bad])
    #print([len(b) for b in bad])

    # DIMS = NUM_IMAGES x MAX_BAD_QUERIES
    dists_bad = np.asarray([clamp(best_dists[i][bad[i]], MAX_BAD_QUERIES) for i in range(len(best_dists))])
    #print([d.shape for d in dists_bad])
    print(dists_bad.shape)
    return dists_bad, d
"""

dir_binary = sys.argv[1]
dir_line = sys.argv[2]

with_simulate = False

print("binary")
dists_bad_binary, _ = load(dir_binary)
print("line")
dists_bad_line, dists_bad_line_sim = load(dir_line)

n = min(dists_bad_line.shape[0], dists_bad_binary.shape[0])
dists_bad_binary = dists_bad_binary[:n]
dists_bad_line = dists_bad_line[:n]
if with_simulate:
    dists_bad_line_sim = dists_bad_line_sim[:n]

#for k in [32, 24, 20, 16, 12, 8]:
#    success_binary = np.mean(dists_bad_binary[:, -1] <= k/255.0)
#    success_line = np.mean(dists_bad_line[:, -1] <= k/255.0)
#    print(k, success_binary, success_line)

plt.figure(figsize=(5, 4))
plt.plot(range(MAX_BAD_QUERIES), np.median(dists_bad_binary, axis=0), label='binary')
plt.plot(range(MAX_BAD_QUERIES), np.median(dists_bad_line, axis=0), label='line (2-stage)')
if with_simulate:
    plt.plot(range(MAX_BAD_QUERIES), np.median(dists_bad_line_sim, axis=0), '--', label='line (1-stage)')
plt.xlabel('Number of bad queries')
plt.ylabel('Median distance')
plt.yscale('log')
plt.legend()
plt.show()
