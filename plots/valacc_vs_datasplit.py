import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# for 0.1 to 0.95, step 0.05
datasplit = np.arange(0.1, 1.0, 0.05)
valaccs = []
datasplits = []
base_dir = "/scratch/grokfast/"
for split in datasplit:
    filename = f"results/res_quad2_split{int(split * 100)}_none_seed0.pt"
    print(f"Checking {filename}")
    if os.path.exists(os.path.join(base_dir, filename)):
        results = torch.load(os.path.join(base_dir, filename))
        # Take average of last 100 epochs
        valacc = np.mean(results['val_acc'][-100:])
        valaccs.append(valacc)
        datasplits.append(split)
    else:
        pass

plt.plot(datasplits, valaccs)
plt.scatter(datasplits, valaccs)
plt.xlabel("Data Split")
plt.ylabel("Validation Accuracy")
plt.ylim(0, 1)
plt.grid()
plt.title("$x^2 + xy + y^2$ mod 97")
plt.savefig(os.path.join(base_dir, "plots/quad1_valacc_vs_datasplit.png"))
