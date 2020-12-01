"""
Train CIFAR10 using SGD. Adapted from https://github.com/kuangliu/pytorch-cifar
Available under the MIT License. Copyright 2017 liukuang
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

from bnn_priors.exp_utils import HDF5Metrics, HDF5ModelSaver, get_data, get_model, he_uniform_initialize


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--model', default="thin_resnet18", type=str, help='name of model')
parser.add_argument('--data', default="cifar10_augmented", type=str, help='name of data set')
parser.add_argument('--width', default=64, type=int, help='width of nn architecture')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
data = get_data(args.data, device=device)
trainloader = torch.utils.data.DataLoader(data.norm.train, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(data.norm.test, batch_size=100, shuffle=False)

# Model
print('==> Building model..')
model = get_model(data.norm.train_X, data.norm.train_y, model=args.model,
                  width=args.width, depth=3,
                  weight_prior="improper", weight_loc=0., weight_scale=1.,
                  bias_prior="improper", bias_loc=0., bias_scale=1.,
                  batchnorm=True, weight_prior_params={}, bias_prior_params={})
model = model.to(device)
print(model)
if device == torch.device('cuda'):
    cudnn.benchmark = True

he_uniform_initialize(model)  # We destroyed He init by using priors, bring it back

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, 0.1)  # Decrease to 1/10 every 150 epochs


# Training
def train(epoch, metrics_saver):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model.net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    metrics_saver.add_scalar("loss", train_loss/(batch_idx+1), epoch)
    metrics_saver.add_scalar("acc", correct/total, epoch)
    print(f"Epoch {epoch}: loss={train_loss/(batch_idx+1)}, acc={correct/total} ({correct}/{total})")
    scheduler.step()


def test(epoch, metrics_saver, model_saver):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    metrics_saver.add_scalar("test/loss", test_loss/(batch_idx+1), epoch)
    metrics_saver.add_scalar("test/acc", correct/total, epoch)
    print(f"Epoch {epoch}: test_loss={test_loss/(batch_idx+1)}, test_acc={correct/total} ({correct}/{total})")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc or (epoch+1)%50 == 0:
        model_saver.add_state_dict(model.state_dict(), epoch)
        model_saver.flush()
        best_acc = acc


with HDF5Metrics("./metrics.h5", "w") as metrics_saver,\
     HDF5ModelSaver("./samples.pt", "w") as model_saver:
    for epoch in range(150*3):
        train(epoch, metrics_saver)
        test(epoch, metrics_saver, model_saver)
