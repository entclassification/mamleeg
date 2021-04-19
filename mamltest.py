import mamltrain
import model
import task
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
wandb.init()

model_save_dir = 'C:/Users/Benton/Desktop/PersonalResearch/mamleeg/'
con = wandb.config
train_task_types = con.tasks[0]
val_task_ids = con.tasks[1]
print(train_task_types)
encoder = model.get_shared_model().to(device)
# best parameters have been with 8 16
t_sampler = task.task_sampler(
    task_batch_size=4, data_batch_size=8, shuffle=True, share=True,
    train_task_types=train_task_types, val_task_ids=val_task_ids)
loss_func = nn.CrossEntropyLoss()
outer_optim = optim.Adam(encoder.parameters(), lr=con.outer_lr, betas=[0.9, 0.999])

wandb.watch(encoder)

for ep in range(con.epochs):
    encoder, train_loss, losses, accs, names = mamltrain.epochtrain(encoder, t_sampler, loss_func, outer_optim, con, ep=ep, plot=True, name=wandb.run.name)
    loss_dict = dict(zip([name + " Loss" for name in names], losses))
    acc_dict = dict(zip([name + " Acc" for name in names], accs))
    mean_dict = {"Val_Loss": np.mean(losses), "Val Acc": np.mean(accs), "Mean Train Loss": train_loss}
    wandb.log({**mean_dict, **loss_dict, **acc_dict})

Path(model_save_dir + wandb.run.name).mkdir(parents=True, exist_ok=True)
torch.save(encoder.state_dict(), model_save_dir + wandb.run.name + '/encoder.pth')
for task_type, classifier in t_sampler.classifiers.items():
    torch.save(classifier.state_dict, model_save_dir + wandb.run.name + '/' + task_type + '.pth')
