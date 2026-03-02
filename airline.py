import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

# Activation functions
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
leaky = np.where(x > 0, x, 0.01 * x)

# Derivatives
sigmoid_d = sigmoid * (1 - sigmoid)
tanh_d = 1 - tanh**2
relu_d = np.where(x > 0, 1, 0)
leaky_d = np.where(x > 0, 1, 0.01)

# Plot
fig, ax = plt.subplots(2, 4, figsize=(12, 6))

ax[0, 0].plot(x, sigmoid)
ax[0, 0].set_title("Sigmoid")
ax[1, 0].plot(x, sigmoid_d)

ax[0, 1].plot(x, tanh)
ax[0, 1].set_title("Tanh")
ax[1, 1].plot(x, tanh_d)

ax[0, 2].plot(x, relu)
ax[0, 2].set_title("ReLU")
ax[1, 2].plot(x, relu_d)

ax[0, 3].plot(x, leaky)
ax[0, 3].set_title("Leaky ReLU")
ax[1, 3].plot(x, leaky_d)

plt.tight_layout()
plt.show()

