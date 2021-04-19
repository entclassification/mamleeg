import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_grad_flow(named_parameters, it, name, ep):

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.25)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    Path('figures/' + name).mkdir(parents=True, exist_ok=True)
    plt.savefig('figures/' + name + '/' + str(it) + '_' + str(ep) + '.png', bbox_inches='tight')


def calc_accuracy(outs, labels):
    num_correct = 0
    total = 0
    for o, labs in zip(outs, labels):
        _, max_indices = torch.max(o, 1)
        num_correct += (max_indices == labs).sum().item()
        total += max_indices.size()[0]

    return float(num_correct) / float(total)

def get_per_step_loss_importance_vector(ep, config):

    loss_weights = np.ones(shape=(config.train_inner_iters)) * (1.0 / config.train_inner_iters)
    decay_rate = 1.0 / config.train_inner_iters / config.epochs
    min_value_for_non_final_losses = 0.03 / config.train_inner_iters
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (ep * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (ep * (config.train_inner_iters - 1) * decay_rate),
        1.0 - ((config.train_inner_iters - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights).to(device)
    return loss_weights

def get_loss_and_accuracy(encoder, task, loss_func, ep, config, importance_weighting=False):

    temp_weights_enc = [w.clone() for w in list(encoder.parameters())]
    weights_classifier = list(task.classifier.parameters())
    if config.use_filter:
        weights_filter = list(task.filter.parameters())
    else:
        weights_filter = []

    losses = []
    final_outs = []
    final_labels = []

    for i, (X_batch, label_batch) in enumerate(task.train(config.train_inner_iters)):
        X_batch = X_batch.to(device)
        label_batch = label_batch.long().to(device)
        if config.use_filter:
            inp = task.filter.parameterised(X_batch, weights_filter)
        else:
            inp = X_batch
        out = task.classifier.parameterised(encoder.parameterised(
            inp, temp_weights_enc), weights_classifier)
        loss = loss_func(out, label_batch)
        grads = torch.autograd.grad(
            loss, weights_filter + temp_weights_enc + weights_classifier, create_graph=True, allow_unused=True)
        filt_grads = grads[:len(weights_filter)]
        enc_grads = grads[len(weights_filter):len(weights_filter) + len(temp_weights_enc)]
        classifier_grads = grads[len(weights_filter) + len(temp_weights_enc):]

        if config.use_filter:
            weights_filter = [w - config.inner_lr * g for w,
                              g in zip(weights_filter, filt_grads)]
        temp_weights_enc = [w - config.inner_lr * g for w,
                            g in zip(temp_weights_enc, enc_grads)]
        weights_classifier = [w - config.inner_lr * g for w,
                              g in zip(weights_classifier, classifier_grads)]

        loss = torch.Tensor([0.0]).to(device)
        outs = []
        labels = []
        for X_batch, label_batch in task.val(config.val_inner_iters):
            X_batch = X_batch.to(device)
            label_batch = label_batch.long().to(device)
            if config.use_filter:
                inp = task.filter.parameterised(X_batch, weights_filter)
            else:
                inp = X_batch
            out = task.classifier.parameterised(encoder.parameterised(
                inp, temp_weights_enc), weights_classifier)
            outs.append(out)
            labels.append(label_batch)
            loss = loss + loss_func(out, label_batch)
        if i == config.train_inner_iters - 1:
            final_outs = outs
            final_labels = labels
        loss = loss / config.val_inner_iters
        losses.append(loss)

    if importance_weighting:
        importance_vec = get_per_step_loss_importance_vector(ep, config)
        sum_losses = torch.Tensor([0.0]).to(device)
        for lo, w in zip(losses, importance_vec):
            sum_losses += lo * w
        loss = torch.mean(sum_losses)
    else:
        loss = losses[-1]

    return loss, calc_accuracy(final_outs, final_labels), classifier_grads


def epochtrain(encoder, task_sampler, loss_func, total_optim, config, ep, plot=False, name=''):
    torch.autograd.set_detect_anomaly(True)
    encoder.train()
    all_train_losses = []
    with torch.set_grad_enabled(True):
        for it, task_batch in enumerate(task_sampler.train_iter()):
            all_loss = torch.Tensor([0.0]).to(device)
            ts = []
            c_all_grads = []
            filters_used = set()
            for task in task_batch:
                task.classifier.train()
                if config.use_filter:
                    filters_used.add(task.filter)
                ts.append(task)
                loss, _, c_gs = get_loss_and_accuracy(encoder, task, loss_func, ep, config, importance_weighting=True)
                c_all_grads.append(c_gs)
                all_loss = all_loss + loss

            all_loss = all_loss / task_sampler.task_batch_size

            if config.use_filter:
                for f in filters_used:
                    filter_grads = torch.autograd.grad(all_loss, f.parameters(), retain_graph=True)
                    with torch.no_grad():
                        for w, g in zip(list(f.parameters()), filter_grads):
                            w.copy_(w.data - config.inner_lr * g.data)

            grads = torch.autograd.grad(all_loss, encoder.parameters())
            all_train_losses.append(all_loss.item())
            for w, g in zip(encoder.parameters(), grads):
                w.grad = g
            # update after all
            total_optim.step()

            # set task classifer weights once not needed
            for t, gs in zip(ts, c_all_grads):
                c_state_dict = t.classifier.state_dict()
                for (n, w), g in zip(c_state_dict.items(), gs):
                    c_state_dict[n].copy_(w - config.inner_lr * g.detach())
            for t in ts:
                if config.use_filter:
                    t.filter.zero_grad()
                t.classifier.zero_grad()
            total_optim.zero_grad()
    encoder.eval()

    val_losses = []
    val_accs = []
    names = []
    for task_batch in task_sampler.val_iter():
        for task in task_batch:
            task.classifier.eval()
            loss, acc, _ = get_loss_and_accuracy(encoder, task, loss_func, ep, config)
            val_losses.append(loss.item())
            val_accs.append(acc)
            names.append(task.name)
            task.classifier.zero_grad()
    encoder.zero_grad()

    return encoder, np.mean(all_train_losses), val_losses, val_accs, names
