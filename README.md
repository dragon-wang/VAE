## VAE generate results
Sample 100 latent codes from normal distribution and input them into the trained decoder:
![generate_from_vae](./generate results/generate_from_vae.png)


## CVAE generate results
Sample 100 latent codes from normal distribution and select the label of which image you want to generate, and then concat them to input into trained decoder:

**The expected label is 1:**

![generate_from_cvae_with_label_1](./generate results/generate_from_cvae_with_label_1.png)

**The expected label is 9:**

![generate_from_cvae_with_label_9](./generate results/generate_from_cvae_with_label_9.png)

## Note

If you  adjust the hyper-parameters carefully (such as the structure of the neural network, the dim of latent or the times of train), the results may be better.