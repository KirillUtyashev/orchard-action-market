import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha0 = 1e-3          # start at 0.001
alpha_min = 1e-6       # floor at 1e-6
T_train = 1_000_000    # total training timesteps (edit as needed)
frac_decay = 0.8       # decay over first 80%

T_decay = int(frac_decay * T_train)
t = np.arange(T_train + 1)

# Linear decay then flat
lr = np.where(
    t <= T_decay,
    alpha0 - (alpha0 - alpha_min) * (t / T_decay),
    alpha_min,
    )

# Plot
plt.figure(figsize=(7, 4))
plt.plot(t, lr, linewidth=2)
plt.xlabel("Timestep")
plt.ylabel("Learning rate")
plt.title("Linear LR decay")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: log-scale y-axis to make the tail more visible

