
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from visdom_def import VisdomPlotter
from PIL import Image
from ait2celeb import A2CDataset
from option import parser

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                ])

train_loader = torch.utils.data.DataLoader(A2CDataset('./ait2celeb', except_folder='trainB, testA', transform=transform),
                                batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(A2CDataset('./ait2celeb', except_folder='trainB, trainA, testB', transform=transform),
                                batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

test_image = transform(Image.open('./ait2celeb/testA/000275.jpg')).view(1, 3, 64, 64)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.convo = nn.Sequential(
            nn.Conv2d(3, 64, 3 ,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 2),
            nn.ReLU(inplace=True),
        )

        self.convotrans = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2),
        )

        # for encoder
        self.fc1 = nn.Linear(1024, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc12 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, 20)
        self.fc22 = nn.Linear(128, 20)

        # for decoder
        self.fc3 = nn.Linear(20, 128)
        self.fc31 = nn.Linear(128, 256)
        self.fc32 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)

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
        h3 = self.fc4(h3)
        h3 = self.convotrans(h3.view(h3.shape[0],h3.shape[-1], 1, 1))
        return torch.sigmoid(h3)

    def forward(self, x):
        x = self.convo(x)
        x = x.view(x.shape[0],1,x.size(1) * x.size(2))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
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
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
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
        python3 -u mainAIT.py > ./checkpoint/log_VAE_AIT.log
"""

global plotter 
global show_image
plotter = VisdomPlotter(env_name='VAE_AIT')

model = VAE().to(device)
model.eval()

if False:
    model = model.load_state_dict(torch.load('./checkpoint/AIT500/model_VAE_AIT.pth'))
    
optimizer = optim.Adam(model.parameters(), lr=1e-4)
save_loss = []
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        reco = model(test_image.to(device))
        save_loss.append([train_loss, test_loss])
        plotter.plot('loss', 'train', 'Class Loss', epoch, train_loss)
        plotter.plot('loss', 'val', 'Class Loss', epoch, test_loss)
        plotter.show_img('sample images', sample[:5])
        plotter.show_img('recon images', reco[0])
        print("save image: " + 'results/sample_' + str(epoch) + '.png')
        save_image(sample, 'results/sample_' + str(epoch) + '.png')
        if epoch % 10 == 0:
            torch.save(save_loss, './checkpoint/loss_VAE_AIT.loss')
            torch.save(model.state_dict(), './checkpoint/model_VAE_AIT.pth')
