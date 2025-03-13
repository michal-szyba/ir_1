import numpy as np


sobel_h = np.array([[-1, 0, 1],
					[-2, 0, 2],
					[-1, 0, 1]])
sobel_v = np.array([[-1, -2, -1],
					[0, 0, 0],
					[1, 2, 1]])


def convolve2d(image, kernel, stride=1, padding=0):
	kernel_height, kernel_width = kernel.shape

	if padding > 0:
		padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
	else:
		padded_image = image

	padded_image_height, padded_image_width = padded_image.shape
	output_height = (padded_image_height - kernel_height) // stride + 1
	output_width = (padded_image_width - kernel_width) // stride + 1

	output = np.zeros((output_height, output_width))

	for y in range(output_height):
		for x in range(output_width):
			region = padded_image[y * stride:y * stride + kernel_height,
					 x * stride:x * stride + kernel_width]
			output[y, x] = np.sum(region * kernel)

	return output


def convolve2dcomplex(image, stride=1, padding=0):
	res_v = convolve2d(image, sobel_v, stride, padding)
	res_h = convolve2d(image, sobel_h, stride, padding)
	res = np.sqrt(res_v**2 + res_h**2)
	return res

def max_pooling(image, pool_size, stride):
	image_height, image_width = image.shape
	output_height = (image_height - pool_size) // stride + 1
	output_width = (image_width - pool_size) // stride + 1
	output = np.zeros((output_height, output_width))
	for y in range(output_height):
		for x in range(output_width):
			region = image[y * stride:y * stride + pool_size, x * stride:x * stride + pool_size]
			output[y, x] = np.max(region)
	return output


