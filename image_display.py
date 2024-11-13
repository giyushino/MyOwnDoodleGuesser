import numpy as np
import matplotlib.pyplot as plt
import torch

#Load in the .npy file
load = np.load(r"classes/full_numpy_bitmap_monkey.npy")

print(load.shape)

# Assuming 'load[100]' is the flattened image
loaded_image = load[400]
print(loaded_image.shape)
# Reshape the flattened vector into a 28x28 image
image = loaded_image.reshape(28, 28)
print(image.shape)
print(image)
# Display the image
plt.imshow(image, cmap='gray')
plt.axis('on')  # Turn off axis for better visualization
plt.show()
