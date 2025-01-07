import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import os


# Homogenity checking
def is_homogeneous(block, threshold=10):
    return np.var(block) <= threshold


# Decomposition recursion
def quadtree_decomposition(image, x, y, width, height, threshold, min_size):
    if (width <= min_size or height <= min_size or
            is_homogeneous(image[y:y+height, x:x+width], threshold)):
        return [(x, y, width, height)]
    half_width = width // 2
    half_height = height // 2
    blocks = []
    # Top left
    blocks += quadtree_decomposition(image, x, y, half_width,
                                     half_height, threshold, min_size)
    # Top right
    if x + half_width < image.shape[1]:
        blocks += quadtree_decomposition(image, x + half_width,
                                         y, width - half_width,
                                         half_height, threshold, min_size)
    # Bottom left
    if y + half_height < image.shape[0]:
        blocks += quadtree_decomposition(image, x, y + half_height,
                                         half_width, height - half_height,
                                         threshold, min_size)
    # Bottom right
    if x + half_width < image.shape[1] and y + half_height < image.shape[0]:
        blocks += quadtree_decomposition(image, x + half_width,
                                         y + half_height, width - half_width,
                                         height - half_height,
                                         threshold, min_size)

    return blocks


# Load grayscale image
image = io.imread(r"./test/sample.jpg", as_gray=True) * 255
image = image.astype(np.uint8)

# Quadtree decomposition parameter
threshold = 20
min_size = 3

# Algorithm execution
blocks = quadtree_decomposition(image, 0, 0, image.shape[1],
                                image.shape[0], threshold, min_size)

# Result output
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
for x, y, width, height in blocks:
    rect = plt.Rectangle((x, y), width, height,
                         edgecolor='red', facecolor='none')
    ax.add_patch(rect)
plt.title('Quadtree Decomposition Result')
plt.show()
