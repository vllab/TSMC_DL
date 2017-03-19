from scipy import misc
import numpy as np

# inverse_transform: transform image value from [-1, 1] to [0, 1]
def inverse_transform(images):
    return (images + 1.) / 2.

# Do inverse_transform before saving the grid image
def save_images(image_path, images, grid_size):
    return imsave(image_path, inverse_transform(images), grid_size)

# Save the grid image
def imsave(image_path, images, grid_size):
    return misc.toimage(merge(images, grid_size), cmin=0, cmax=1).save(image_path)

# merge images to a grid image
def merge(images, grid_size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) == 3:  #batch, row, col
        c = 1
        img = np.zeros((h * grid_size[0], w * grid_size[1]))
    else:
        c = images.shape[3]
        img = np.zeros((h * grid_size[0], w * grid_size[1], c))

    for idx, image in enumerate(images):
        i = idx % grid_size[0]
        j = idx // grid_size[0]
        if c == 1:
            img[i*w:i*w+w, j*h:j*h+h] = image
        else:
            img[i*w:i*w+w, j*h:j*h+h, :] = image
    return img
