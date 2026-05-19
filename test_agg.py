import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
print("matplotlib OK, backend:", matplotlib.get_backend())

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
fig.savefig("test_fig.png", dpi=150)
plt.close()
print("Figure saved OK")
