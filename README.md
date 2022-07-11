# compress-multiphoton
Compression for multiphoton calcium imaging movies and stacks

## Grayscale quantization

The first step for image compression is to enforce proper grayscale quantization of the images.
Most two-photon acquisition systems, in our experience, are quantized with excessive grayscale resolution so that the least significant bits encode only noise. 

We can requantize images using fewer bits  while avoiding introducing biases or extra variance.
