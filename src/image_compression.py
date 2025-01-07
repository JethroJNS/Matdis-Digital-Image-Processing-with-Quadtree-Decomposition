import numpy as np
from skimage import io, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import os


# Homogeneity checking
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


# Compress image with quadtree decomposition
def compress_with_quadtree(image, blocks):
    compressed = np.zeros_like(image)
    for x, y, width, height in blocks:
        mean_value = np.mean(image[y:y+height, x:x+width])
        compressed[y:y+height, x:x+width] = mean_value
    return compressed


# Compress image without quadtree decomposition (simple downsampling)
def compress_without_quadtree(image, downscale_factor):
    small = cv2.resize(
        image,
        (
            image.shape[1] // downscale_factor,
            image.shape[0] // downscale_factor,
        ),
        interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_AREA)


# Calculate the effective compression ratio of quadtree
def calculate_quadtree_compression_ratio(image, blocks):
    total_blocks = sum([w * h for _, _, w, h in blocks])
    original_pixels = image.size
    return original_pixels / total_blocks


# Load grayscale image
image = io.imread(r"./test/sample.jpg",
                  as_gray=True) * 255
image = image.astype(np.uint8)

# Quadtree decomposition parameters
threshold = 20
min_size = 3

# Execute quadtree decomposition
blocks = quadtree_decomposition(image, 0, 0, image.shape[1],
                                image.shape[0], threshold, min_size)
compressed_quadtree = compress_with_quadtree(image, blocks)

# Calculate the compression ratio of quadtree
compression_ratio_quadtree = calculate_quadtree_compression_ratio(image,
                                                                  blocks)

# Determine downscale factor for similar compression ratio
downscale_factor = int(np.sqrt(image.size / (image.size /
                                             compression_ratio_quadtree)))

# Execute compression without quadtree decomposition
compressed_no_quadtree = compress_without_quadtree(image, downscale_factor)

# Compute SSIM
ssim_quadtree = ssim(image, compressed_quadtree)
ssim_no_quadtree = ssim(image, compressed_no_quadtree)

# Results output
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(compressed_quadtree, cmap='gray')
axes[1].set_title(f'Compressed (Quadtree)\nSSIM: {ssim_quadtree:.4f}')
axes[1].axis('off')

axes[2].imshow(compressed_no_quadtree, cmap='gray')
axes[2].set_title(f'Compressed (No Quadtree)\nSSIM: {ssim_no_quadtree:.4f}')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Save compressed images
io.imsave(r"./test/compressed_quadtree.jpg",
          img_as_ubyte(compressed_quadtree))
io.imsave(r"./test/compressed_no_quadtree.jpg",
          img_as_ubyte(compressed_no_quadtree))
