import numpy as np
import matplotlib.pyplot as plt
import random

def show_image():
    print("crab, crocodile, lion, lobster, octopus, panda, swan")
    while True:
        while True:
            ask = input("What animal would you like to see? ")
            try:
                loading = np.load(rf"classes/full_numpy_bitmap_{ask}.npy")
                break
            except FileNotFoundError:
                print("Oops! Invalid class")
        loaded_image = loading[random.randint(0, len(loading) - 1)]
        image = loaded_image.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(ask)
        plt.axis()  # Turn off axis for better visualization
        plt.show()

show_image()
