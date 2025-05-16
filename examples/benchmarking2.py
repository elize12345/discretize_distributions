import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import torch
import pandas as pd
import time
import discretize_distributions as dd
from shelling_gmms import dbscan_shells, quantization_gmm_shells

mean_distances = np.linspace(0.5, 8.0, 6)
results = defaultdict(list)
torch.manual_seed(1)
file = []

for d in mean_distances:
    print(f"--- Testing with component distance d = {d:.2f} ---")

    locs = torch.tensor([[-d / 2, 0.0], [d / 2, 0.0]], dtype=torch.float32)
    probs = torch.tensor([0.5, 0.5], dtype=torch.float32)
    cov = torch.eye(2).unsqueeze(0).repeat(2, 1, 1)  # identity for each component

    test_case = {
        "locs": locs,
        "covariance_matrix": cov,
        "probs": probs
    }

    num_dims = locs.shape[1]
    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=cov * (1 / (np.sqrt(num_dims)))  # optional scaling
        )
    )

    # --- W2-based discretization ---
    start = time.time()
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    locs_gmm = disc_gmm.locs
    sig_runtime = time.time() - start
    sig_w2 = disc_gmm.w2.item()
    sig_num_locs = len(locs_gmm)
    print(f"Total W2 error w2-optimal: {sig_w2}")
    print(f"Total number of locations w2-optimal: {len(locs_gmm)}")

    # --- grid-based discretization ---
    start = time.time()
    shells, z = dbscan_shells(gmm)
    probs_, locs_, shell_w2, grids = quantization_gmm_shells(
        gmm, shells, z, paddings=[0.01] * len(shells)
    )
    shell_runtime = time.time() - start
    shell_num_locs = len(locs_)
    nr_shells = len(shells)
    print(f"Total W2 error over all grids: {shell_w2}")
    print(f"Total number of locations grids: {len(locs_)}")

    # --- metrics ---
    results["d"].append(d)
    results["sig_w2"].append(sig_w2)
    results["sig_num_locs"].append(sig_num_locs)
    results["sig_runtime"].append(sig_runtime)
    results["shell_w2"].append(shell_w2)
    results["shell_num_locs"].append(shell_num_locs)
    results["shell_runtime"].append(shell_runtime)

    file.append({
        "d": d,
        "sig_w2": sig_w2,
        "sig_num_locs": sig_num_locs,
        "sig_runtime": sig_runtime,
        "shell_w2": shell_w2,
        "shell_num_locs": shell_num_locs,
        "shell_runtime": shell_runtime,
        "nr of shells": nr_shells,
    })

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.scatter(locs_gmm.detach().numpy()[:, 0], locs_gmm.detach().numpy()[:, 1],
                label="Mixture of W2-optimal locs", color='blue', marker='o', alpha=0.3)
    ax.scatter(locs_[:, 0], locs_[:, 1], label="Mixture of grids locs", color='red', alpha=0.6)

    cmap = plt.colormaps.get_cmap('tab10')
    for idx, R1 in enumerate(grids):
        core_lower_vertices_per_dim = R1.lower_vertices_per_dim
        core_upper_vertices_per_dim = R1.upper_vertices_per_dim
        for i in range(len(core_lower_vertices_per_dim[0])):
            for j in range(len(core_lower_vertices_per_dim[1])):
                x0 = core_lower_vertices_per_dim[0][i].item()
                x1 = core_upper_vertices_per_dim[0][i].item()
                y0 = core_lower_vertices_per_dim[1][j].item()
                y1 = core_upper_vertices_per_dim[1][j].item()
                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor='blue',
                    facecolor='none',
                    linewidth=1.5,
                    linestyle='-'
                )
                ax.add_patch(rect)

    plt.title(f"Locs of discretization for component distance d={d:.2f} ")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    # plt.savefig(f'Benchmark_tests/2D_GMMs/locs_discretization_for_{d}.svg')
    plt.show()

# d values
d_vals = results["d"]

# saving
df = pd.DataFrame(results)
df.to_csv("Benchmark_tests/2D_GMMs/results.csv", index=False)
print("Experiment complete. Results saved to csv file.")

plt.figure()
plt.plot(d_vals, results["sig_w2"], label="Mixture of W2-optimal", marker="o")
plt.plot(d_vals, results["shell_w2"], label="Mixture of grids", marker="x")
plt.xlabel("Component distance (d) from means")
plt.ylabel("W2 error")
plt.title("W2 error vs component distance (for 2D GMMs)")
plt.legend()
plt.grid(True)
# plt.savefig(f'Benchmark_tests/2D_GMMs/w2_error.svg')
plt.show()

plt.figure()
plt.plot(d_vals, results["sig_num_locs"], label="Mixture of W2-optimal", marker="o")
plt.plot(d_vals, results["shell_num_locs"], label="Mixture of grids", marker="x")
plt.xlabel("Component distance (d) from means")
plt.ylabel("Size of support")
plt.title("Support size vs component distance (for 2D GMMs)")
plt.legend()
plt.grid(True)
# plt.savefig(f'Benchmark_tests/2D_GMMs/support.svg')
plt.show()

plt.figure()
plt.plot(results["sig_num_locs"], results["sig_w2"], label="Mixture of W2-optimal", marker="o")
plt.plot(results["shell_num_locs"], results["shell_w2"], label="Mixture of grids", marker="x")
plt.xlabel("Size of support")
plt.ylabel("W2 error")
plt.title("Trade-off: W2 error vs support size (for 2D GMMs)")
plt.legend()
plt.grid(True)
# plt.savefig(f'Benchmark_tests/2D_GMMs/trade_off.svg')
plt.show()
