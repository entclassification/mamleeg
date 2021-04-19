import torch
import torch.nn as nn
import torch.nn.functional as F

conv_init = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    global conv_init
    if isinstance(m, nn.Conv2d):
        if conv_init is None:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            conv_init(m.weight)


F1 = 8
D = 4
F2 = D * F1
C = 22
p = 0.5
T = 128


class ConstrainedConv2d(nn.Module):
    def __init__(self, nin, nout, kernel, groups, padding=0):
        super(ConstrainedConv2d, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel, padding=padding, groups=groups)

    def forward(self, x):
        return F.conv2d(x, self.conv.weight.clamp(min=-1.0, max=1.0), self.conv.bias.clamp(min=-1.0, max=1.0), self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

    def parameterised(self, x, weight):
        return F.conv2d(x, weight[0].clamp(min=-1.0, max=1.0), weight[1].clamp(min=-1.0, max=1.0), self.conv.stride, self.conv.padding, groups=self.conv.groups)


class ConstrainedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias.clamp(min=-1.0, max=1.0))

    def parameterised(self, x, weight):
        return F.linear(x, weight[0].clamp(min=-1.0, max=1.0), weight[1].clamp(min=-1.0, max=1.0))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def parameterised(self, x, weights):
        x = F.conv2d(x, weights[0], weights[1], self.depthwise.stride,
                     self.depthwise.padding, groups=self.depthwise.groups)
        x = F.conv2d(x, weights[2], weights[3], self.pointwise.stride,
                     self.pointwise.padding, groups=self.pointwise.groups)
        return x


class Shared(nn.Module):

    def __init__(self):
        super(Shared, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, F1, (1, 64)),
                                    nn.ZeroPad2d((32, 31, 0, 0)),
                                    nn.BatchNorm2d(F1))
        # Layer 2
        self.layer2 = nn.Sequential(ConstrainedConv2d(F1, D * F1, (C, 1), groups=F1),
                                    nn.BatchNorm2d(D * F1),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 4)),
                                    nn.Dropout(p=p))

        self.layer3 = nn.Sequential(DepthwiseSeparableConv(D * F1, F2, (1, 16)),
                                    nn.ZeroPad2d((8, 7, 0, 0)),
                                    nn.BatchNorm2d(F2),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 8)),
                                    nn.Dropout(p=p))

    def forward(self, x):
        # dont permute if spatial filter first
        # permute so that ensemble of inputs are in first dim
        x = x.permute(1, 0, 2, 3, 4)
        out = torch.Tensor([0.0]).to(device)
        for row in x:
            row = self.layer1(row)
            row = self.layer2(row)
            row = self.layer3(row)
            row = torch.flatten(row, start_dim=1)
            out = out + row
        out = out / x.shape[0]
        return out

    def parameterised(self, x, weights, bn_training=True):
        idx = 0
        # dont permute if spatial filter first
        x = x.permute(0, 3, 1, 2)
        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                x = F.conv2d(x, weights[idx], weights[idx + 1],
                             m.stride, m.padding, groups=m.groups)
                idx += 2

            elif isinstance(m, nn.BatchNorm2d):
                x = F.batch_norm(x, m.running_mean, m.running_var,
                                 weights[idx], weights[idx + 1], bn_training)
                idx += 2
            elif isinstance(m, nn.ReLU):
                x = F.relu(x)
            elif isinstance(m, nn.AvgPool2d):
                x = F.avg_pool2d(x, m.kernel_size, m.stride, m.padding)
            elif isinstance(m, nn.ELU) or isinstance(m, nn.Dropout) or isinstance(m, nn.ZeroPad2d):
                x = m(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Classifier(nn.Module):
    def __init__(self, nin, nout):
        super(Classifier, self).__init__()
        self.lin = nn.Linear(nin, nout)

    def forward(self, x):
        x = self.lin(x)
        return x

    def parameterised(self, x, weights):
        x = F.linear(x, weights[0], weights[1])
        return x


class SpatialFilter(nn.Module):
    def __init__(self, nin, nout, kernel, groups=1):
        super(SpatialFilter, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel, groups=groups)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.conv2d(x, self.conv.weight.clamp(min=-1.0, max=1.0), self.conv.bias.clamp(min=-1.0, max=1.0), self.conv.stride,
                     self.conv.padding, self.conv.dilation, self.conv.groups)
        return x

    def parameterised(self, x, weight):
        x = x.permute(0, 3, 1, 2)
        x = F.conv2d(x, weight[0].clamp(min=-1.0, max=1.0), weight[1].clamp(
            min=-1.0, max=1.0), self.conv.stride, self.conv.padding, groups=self.conv.groups)
        print(x.shape)
        return x


class LengthFilter(nn.Module):
    def __init__(self, channels, inp_len, out_len, grouped=False):
        super(LengthFilter, self).__init__()
        if grouped:
            groups = channels
        else:
            groups = 1
        # kernel_length = inp_len - out_len
        kernel_length = int(inp_len / 2)
        dif_length = out_len - (inp_len - (kernel_length - 1))
        if dif_length < 0:
            kernel_length += abs(dif_length)

        self.filter = ConstrainedConv2d(1, channels, kernel=(1, kernel_length), groups=groups)
        pad_length = out_len - (inp_len - (kernel_length - 1))
        if pad_length % 2:
            left_pad = int(pad_length / 2) + 1
            right_pad = int(pad_length / 2)
        else:
            left_pad = int(pad_length / 2)
            right_pad = int(pad_length / 2)
        self.padding = nn.ZeroPad2d((left_pad, right_pad, 0, 0))

    def forward(self, x):
        # Assume input shape (N x C x inp_len x 1)
        # Return shape (N x C x out_len x 1)
        x = self.filter(x)
        x = self.padding(x)
        return x

    def parameterised(self, x):
        x = self.filter.parameterised(x)
        x = self.padding(x)
        return x


class Filter(nn.Module):
    def __init__(self, inp_channels, inp_len, out_len):
        super(Filter, self).__init__()
        self.len_filter = nn.Sequential(LengthFilter(F1, inp_len, out_len),
                                        nn.BatchNorm2d(F1))
        self.spat_filter = nn.Sequential(ConstrainedConv2d(F1, D * F1, (inp_channels, 1), groups=F1),
                                         nn.BatchNorm2d(D * F1),
                                         nn.ELU(),
                                         nn.AvgPool2d((1, 4)),
                                         nn.Dropout(p=p))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.len_filter(x)
        x = self.spat_filter(x)

        return x

    def parameterised(self, x, weights, bn_training=True):
        idx = 0
        # dont permute if spatial filter first
        x = x.permute(0, 3, 1, 2)
        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                x = F.conv2d(x, weights[idx], weights[idx + 1],
                             m.stride, m.padding, groups=m.groups)
                idx += 2

            elif isinstance(m, nn.BatchNorm2d):
                x = F.batch_norm(x, m.running_mean, m.running_var,
                                 weights[idx], weights[idx + 1], bn_training)
                idx += 2
            elif isinstance(m, nn.ReLU):
                x = F.relu(x)
            elif isinstance(m, nn.AvgPool2d):
                x = F.avg_pool2d(x, m.kernel_size, m.stride, m.padding)
            elif isinstance(m, nn.ELU) or isinstance(m, nn.Dropout) or isinstance(m, nn.ZeroPad2d):
                x = m(x)
        return x


def get_classifier(num_classes):
    return Classifier(int((F2 * T) / 32), num_classes)


def get_shared_model(conv_type_init=None):
    global conv_init
    model = Shared()
    if conv_type_init == 'uniform':
        conv_init = torch.nn.init.uniform_
    elif conv_type_init == 'xavier_uniform':
        conv_init = torch.nn.init.xavier_uniform_
    elif conv_type_init == 'xavier_normal':
        conv_init = torch.nn.init.xavier_normal_
    elif conv_type_init == 'normal':
        conv_init = torch.nn.init.normal_
    if conv_type_init is not None:
        model.apply(weights_init)

    return model
