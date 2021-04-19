import task
import model
import mamltrain
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import argparse
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('path')
parser.add_argument('lr')
parser.add_argument('mom')
parser.add_argument('weight_init')
parser.add_argument('repeat')
args = parser.parse_args()
path = args.path
exp_name = path[path.rfind('\\') + 1:]
exp_id = exp_name[:2]
subjectid = exp_name[3]
task_type = None

single_task = task.competetask(path, 16)

if path.find("HI") != -1:
    task_type = "SGLHandHI"
elif path.find("SGLHand") != -1:
    task_type = "SGLHand"
elif path.find("LRHand") != -1:
    task_type = "LRHand"
elif path.find("Tongue") != -1:
    task_type = "Tongue"
elif path.find("NoMotor") != -1:
    task_type = "NoMotor"
elif path.find("Comp") != -1:
    task_type = "Comp"
    subjectid += '\''
else:
    print("Cannot find task type in ", path)

task_type = str(single_task.stats['num_classes']) + 'St' + task_type

config = {"subjectid": subjectid, "task_type": task_type, "num_classes": single_task.stats['num_classes'],
          "lr": float(args.lr), "mom": float(args.mom), "weight_init": args.weight_init}

name = exp_id + '-' + subjectid + "-" + task_type + str(args.repeat)

wandb.init(project='GroupedCEEG', name=name, config=config)

encoder = model.get_shared_model().to(device)

loss_func = nn.CrossEntropyLoss()

combined_model = nn.Sequential(encoder, single_task.classifier)
total_optim = optim.SGD(combined_model.parameters(), lr=0.01, momentum=0.01)

for ep in range(400):

    combined_model.train()
    train_losses = []
    train_outs = []
    train_labels = []
    for X, y in single_task.train():
        outs = torch.Tensor()
        X = X.to(device)
        y = y.long().to(device)
        total_optim.zero_grad()
        with torch.set_grad_enabled(True):
            out = combined_model(X)
            loss = loss_func(out, y)
            loss.backward()
            total_optim.step()
        train_losses.append(loss.item())
        train_outs.append(out)
        train_labels.append(y)
    train_acc = mamltrain.calc_accuracy(train_outs, train_labels)
    val_losses = []
    val_outs = []
    val_labels = []
    combined_model.eval()
    for X, y in single_task.val():
        X = X.to(device)
        y = y.long().to(device)

        with torch.set_grad_enabled(False):
            out = combined_model(X)
            loss = loss_func(out, y)
        val_losses.append(loss.item())
        val_outs.append(out)
        val_labels.append(y)
    val_acc = mamltrain.calc_accuracy(val_outs, val_labels)
    wandb.log({"Val_Acc": val_acc, "Val_Loss": np.mean(val_losses), "Train_Loss": np.mean(train_losses), "Train_Acc": train_acc})
