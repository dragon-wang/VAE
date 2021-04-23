import torch
import torchvision
from vae_on_mnist import VAE
from cvae_on_mnist import CVAE
import utils


def generate_from_vae(sample_num, latent_dim):
    # Load vae model
    vae = VAE()
    utils.load_model(vae, "./model_weights/vae/vae_weights.pth")

    # sample from the latent space
    z = torch.randn(sample_num, latent_dim)
    recon = vae.decode(z).detach()

    # transfer recon to img
    img = utils.to_img(recon)

    # save img
    torchvision.utils.save_image(img, "./img/vae/generate_from_vae.png", nrow=10)


def generate_from_cvae(sample_num: int, latent_dim: int, label: int):
    # Load vae model
    cvae = CVAE(784, 10, 10)
    utils.load_model(cvae, "./model_weights/cvae/cvae_weights.pth")

    # sample from the latent space and concat label that you want to generate
    z = torch.randn(sample_num, latent_dim)
    labels = torch.full(size=(sample_num, 1), fill_value=label, dtype=torch.int64)
    y = utils.to_one_hot(labels, num_class=10)
    recon = cvae.decode(z, y).detach()

    # transfer recon to img
    img = utils.to_img(recon)

    # save img
    torchvision.utils.save_image(img, "./img/cvae/generate_from_vae.png", nrow=10)


if __name__ == '__main__':
    # generate_from_vae(sample_num=100, latent_dim=10)
    generate_from_cvae(sample_num=100, latent_dim=10, label=9)