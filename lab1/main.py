import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

import os

# ------------------------------------------------------- #
# BUILD DATASET
def f(x):
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    return torch.sin(x) + torch.cos(x) + torch.sin(x) * torch.cos(x)

def build_dataset(num_samples, split_ratio=(0.6, 0.2), seed=0, device="cuda"):
    """ Build the dataset following the lab requirements below:
    在 [0, 2PI) 范围内随机sample x，并计算 y = sin(x) + cos(x) + sin(x)cos(x) 作为 y 值。

    Args:
        num_samples (int): sample number in the total dataset
        seed (int): random seed
        device: "cuda" or "cpu". Defaults to "cuda". 

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
    """

    # set random seed for reproducibility
    torch.manual_seed(seed)
    
    # randomly generate x values [0, 2PI)
    x = torch.rand(num_samples) * 2 * np.pi
    # x = torch.linspace(0, 4 * torch.acos(torch.zeros(1)).item(), num_samples)
    
    # calculate y values
    y = f(x)
    
    # normalize x
    mean = x.mean()
    std = x.std()
    x = (x - x.mean()) / x.std()
    
    # shuffle the data
    indices = torch.randperm(len(x))
    x = x[indices]
    y = y[indices]
    
    # split the data into train, validation, and test sets
    train_size = int(split_ratio[0] * len(x))
    val_size = int(split_ratio[1] * len(x))
    
    x_train = x[:train_size].reshape(-1, 1).to(device)
    y_train = y[:train_size].reshape(-1, 1).to(device)
    
    x_val = x[train_size:train_size + val_size].reshape(-1, 1).to(device)
    y_val = y[train_size:train_size + val_size].reshape(-1, 1).to(device)
    
    x_test = x[train_size + val_size:].reshape(-1, 1).to(device)
    y_test = y[train_size + val_size:].reshape(-1, 1).to(device)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, mean.item(), std.item()


# ------------------------------------------------------- #
# BUILD NEURAL NETWORK
class net(nn.Module):
    def __init__(self, in_chans: int, depth: int, embed_dims: list, act: nn.Module, norm: bool = True, device="cuda"):
        """ Build the neural network, which could manually modify its
        depth, embed_dims, activation functions and whether has residual paths

        Args:
            in_chans (int): e.g. 5
            depth (int): e.g. 5
            embed_dims (list): e.g. [5, 10, 100, 10, 1]
            act (nn.Module): e.g. nn.ReLU
            norm (bool, optional): whether has norm layer. Defaults to True.
            device: "cuda" or "cpu". Defaults to "cuda". 

        """
        super(net, self).__init__()
        
        # restrcitions
        assert embed_dims[-1] == 1 # output: y
        assert embed_dims[0] == in_chans # input: [x, x^2, ..., x^{in_chans}]
        assert len(embed_dims) == depth
        
        self.in_chans = in_chans

        # neural network
        self.layers = nn.ModuleList()
        for i in range(depth-1):
            self.layers.append(nn.Linear(embed_dims[i], embed_dims[i+1]))
            self.layers.append(act)
            if norm:
                self.layers.append(nn.BatchNorm1d(embed_dims[i+1]))
        
        # init and move to device
        self.init_weights()
        self.to(device)
        
    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = torch.cat([x ** i for i in range(1, self.in_chans+1)], dim=1)
        
        for layer in self.layers:
            x = layer(x)

        return x


# ------------------------------------------------------- #
# MAIN
if __name__ == "__main__":
    
    # build dataset
    x_train, y_train, x_val, y_val, x_test, y_test, mean, std = build_dataset(num_samples=5000, split_ratio=(0.6, 0.2), seed=0, device="cuda")
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    
    # build neural network
    model = net(in_chans=2, depth=5, embed_dims=[2, 10, 100, 10, 1], act=nn.ReLU(), norm=True, device="cuda")
    
    # MSE loss
    criterion = nn.MSELoss()
    
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5)

    # training
    print("*"*50)
    print("Start Training")
    print("*"*50)
    train_losses = []
    val_losses = []
    num_epochs = 200
    for epoch in range(num_epochs):
        for idx, (samples, labels) in enumerate(train_loader):

            # clear grad
            optimizer.zero_grad()
            
            # forward, backward, optimize
            y_pred = model(samples)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            
        # lr scheduler
        with torch.no_grad():
            y_train_pred = model(x_train)
            train_loss = criterion(y_train_pred, y_train)
            y_val_pred = model(x_val)
            val_loss = criterion(y_val_pred, y_val)
        scheduler.step(val_loss)
        
        # record for drawing loss curve
        train_losses.append(torch.log(train_loss).item())
        val_losses.append(torch.log(val_loss).item())

        # print info
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            if lr < 1e-4:
                print("Early stop")
                print(f"Final epoch {epoch}, train loss: {train_loss.item()}, val loss: {val_loss.item()}, lr: {lr}\n")
                break
            print(f"Epoch {epoch}, train loss: {train_loss.item()}, val loss: {val_loss.item()}, lr: {lr}")

    # evaluate on dataset
    print("*"*50)
    print("MSE Loss on Dataset")
    print("*"*50)
    with torch.no_grad():
        y_train_pred = model(x_train)
        train_loss = criterion(y_train_pred, y_train)
        y_val_pred = model(x_val)
        val_loss = criterion(y_val_pred, y_val)
        y_test_pred = model(x_test)
        test_loss = criterion(y_test_pred, y_test)
    print(f"Train loss: {train_loss.item()}")
    print(f"Val loss: {val_loss.item()}")
    print(f"Test loss: {test_loss.item()}\n")
    
    # large scale test on 10000 points from [0, 2PI)
    print("*"*50)
    print("Large Scale Test")
    print("*"*50)
    lin_x = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1).to("cuda")
    lin_y = f(lin_x)
    lin_x_n = (lin_x - mean) / std

    pred_lin_y = model(lin_x_n)
    mse_loss = (pred_lin_y - lin_y) ** 2
    print(f"For 10000 points in [0, 2PI): ")
    print(f"The biggest loss is {torch.max(mse_loss).item()}, the smallest loss is {torch.min(mse_loss).item()}")
    print(f"The average loss is {torch.mean(mse_loss).item()}\n")
    
    plt.xlabel("x")
    plt.ylabel("log MSE loss")
    plt.plot(lin_x.detach().cpu().numpy(), torch.log(mse_loss).detach().cpu().numpy())
    plt.savefig("large_scale.png")
    plt.cla()
    
    # draw loss curve
    print("*"*50)
    print("Draw Loss Curve")
    print("*"*50)
    sns.set(rc={'figure.figsize':(10, 6)})
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Log MSE Loss")
    sns.lineplot(x=range(len(train_losses)), y=train_losses, label="train")
    sns.lineplot(x=range(len(val_losses)), y=val_losses, label="val")
    plt.savefig("loss_curve.png")
    dir_name = os.path.dirname(os.path.abspath(__file__))
    print("Loss curve saved to", os.path.join(dir_name, "loss_curve.png"))