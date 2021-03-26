from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from visdom_def import VisdomPlotter
from option import parser

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

out_dir = '../Lab07/torch_data/VGAN/MNIST/dataset' #you can use old downloaded dataset, I use from VGAN

train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, download=True, train=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # for encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc12 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, 20)
        self.fc22 = nn.Linear(128, 20)

        # for decoder
        self.fc3 = nn.Linear(20, 128)
        self.fc31 = nn.Linear(128, 256)
        self.fc32 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 784)

    def encode(self, x):
        x = self.fc1(x)
        x = self.fc11(x)
        h1 = F.relu(self.fc12(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 0.5 for square root (variance to standard deviation)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.fc3(z)
        z = self.fc31(z)
        h3 = F.relu(self.fc32(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
    return train_loss/len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss

"""
Open visdom port:
        python3 -m visdom.server
Train a model:
        unset http_proxy
        unset https_proxy
        python3 -u mainMNIST.py > ./checkpoint/log_VAE_MNIST.log
"""

global plotter 
global show_image
plotter = VisdomPlotter(env_name='VAE_MNIST')

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
save_loss = []
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_loss.append([train_loss, test_loss])
        plotter.plot('loss', 'train', 'Class Loss', epoch, train_loss)
        plotter.plot('loss', 'val', 'Class Loss', epoch, test_loss)
        # label = 'image in epoch' + str(epoch)
        plotter.show_img('recon images', sample.view(64, 1, 28, 28)[:5])
        print("save image: " + 'results/sample_' + str(epoch) + '.png')
        save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
        if epoch % 10 == 0:
            torch.save(save_loss, './checkpoint/loss_VAE_MNIST.loss')
            torch.save(model, './checkpoint/model_VAE_MNIST.pth')