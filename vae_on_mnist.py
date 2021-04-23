import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import utils


class VAE(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2_mu = nn.Linear(200, 10)
        self.fc2_log_std = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 200)
        self.fc4 = nn.Linear(200, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon = torch.sigmoid(self.fc4(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss


if __name__ == '__main__':
    epochs = 100
    batch_size = 100

    recon = None
    img = None

    utils.make_dir("./img/vae")
    utils.make_dir("./model_weights/vae")

    train_data = torchvision.datasets.MNIST(
        root='./mnist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    data_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    vae = VAE()

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(100):
        train_loss = 0
        i = 0
        for batch_id, data in enumerate(data_loader):
            img, _ = data
            inputs = img.reshape(img.shape[0], -1)
            recon, mu, log_std = vae(inputs)
            loss = vae.loss_function(recon, inputs, mu, log_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 100 == 0:
                print("Epoch[{}/{}], Batch[{}/{}], batch_loss:{:.6f}".format(
                    epoch+1, epochs, batch_id+1, len(data_loader), loss.item()))

        print("======>epoch:{},\t epoch_average_batch_loss:{:.6f}============".format(epoch+1, train_loss/i), "\n")

        # save imgs
        if epoch % 10 == 0:
            imgs = utils.to_img(recon.detach())
            path = "./img/vae/epoch{}.png".format(epoch+1)
            torchvision.utils.save_image(imgs, path, nrow=10)
            print("save:", path, "\n")

    torchvision.utils.save_image(img, "./img/vae/raw.png", nrow=10)
    print("save raw image:./img/vae/raw/png", "\n")

    # save val model
    utils.save_model(vae, "./model_weights/vae/vae_weights.pth")


