import torch
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import torchvision.transforms as t
import torch.nn as nn
import model

TRAIN_SESSIONS = 2
TOTAL_SESSIONS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class dataset_(Dataset):

    def __init__(self, paths):
        self.data_paths = []
        if isinstance(paths, list):
            for s in paths:
                self.data_paths += glob.glob(s + '/*')

        else:
            self.data_paths = glob.glob(paths + '/*')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        p = self.data_paths[index]
        with open(p, 'rb') as handle:
            x, lab = pickle.load(handle)
        x = torch.from_numpy(x).float()
        if len(x.shape) == 2:
            x = x.view(1, 1, x.shape[0], x.shape[1])
        else:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        return x, lab


class task():
    def __init__(self, path, batch_size):
        self.batch_size = batch_size
        self.paths = glob.glob(path + '/*')

    def val(self, n=None):
        if n is None:
            while True:
                try:
                    data, target = next(self.valloader_iterator)
                except StopIteration:
                    dset = dataset_(self.val_paths)
                    self.valloader_iterator = iter(DataLoader(
                        dset, batch_size=self.batch_size, shuffle=True))
                    break
                yield data, target

        else:
            for _ in range(n):
                try:
                    data, target = next(self.valloader_iterator)
                except StopIteration:
                    dset = dataset_(self.val_paths)
                    self.valloader_iterator = iter(DataLoader(
                        dset, batch_size=self.batch_size, shuffle=True))
                    data, target = next(self.valloader_iterator)
                yield data, target

    def train(self, n=None):
        if n is None:
            while True:
                try:
                    data, target = next(self.trainloader_iterator)
                except StopIteration:
                    dset = dataset_(self.train_paths)
                    self.trainloader_iterator = iter(DataLoader(
                        dset, batch_size=self.batch_size, shuffle=True))
                    break
                yield data, target
        else:

            for _ in range(n):
                try:
                    data, target = next(self.trainloader_iterator)
                except StopIteration:
                    dset = dataset_(self.train_paths)
                    self.trainloader_iterator = iter(DataLoader(
                        dset, batch_size=self.batch_size, shuffle=True))
                    data, target = next(self.trainloader_iterator)
                yield data, target


class competetask(task):
    def __init__(self, path, batch_size):
        super().__init__(path, batch_size)
        self.batch_size = batch_size
        self.train_paths = self.paths[0]
        self.stats = {'num_classes': 4, 'num_channels': 22, 'length': 250}
        self.val_paths = self.paths[1]
        self.name = path[path.rfind('\\') + 4] + '\'' + 'Comp'
        self.classifier = model.get_classifier(4).to(device)
        self.classifier_id = 'Comp'
        self.filter_id = 'Comp'
        self.train_dataset = dataset_(self.train_paths)
        self.val_dataset = dataset_(self.val_paths)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True)

        self.trainloader_iterator = iter(self.train_dataloader)
        self.valloader_iterator = iter(self.val_dataloader)


class largetask(task):
    def __init__(self, path, batch_size):
        super().__init__(path, batch_size)
        self.session_paths = []
        self.stats = None
        self.stat_list = []
        for p in self.paths:
            if "Session" in p:
                self.session_paths.append(p)
            elif "stats" in p:
                with open(p, 'rb') as handle:
                    self.stat_list.append(pickle.load(handle))
                if len(self.stat_list) == 0:
                    print("could not find stats!")
        self.set_stats()
        self.set_name(path)
        self.classifier = model.get_classifier(
            int(self.stats['num_classes'])).to(device)
        self.session_paths = np.random.permutation(self.session_paths).tolist()
        self.train_paths = self.session_paths[:TRAIN_SESSIONS]
        self.val_paths = self.session_paths[TRAIN_SESSIONS:]

        self.train_dataset = dataset_(self.train_paths)
        self.val_dataset = dataset_(self.val_paths)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True)

        self.trainloader_iterator = iter(self.train_dataloader)
        self.valloader_iterator = iter(self.val_dataloader)

    def set_name(self, path):
        if path.find("HI") != -1:
            task_type = "SGLHAND-HI"
        if path.find("SGLHand") != -1:
            task_type = "SGLHand"
        elif path.find("LRHand") != -1:
            task_type = "LRHand"
        elif path.find("Tongue") != -1:
            task_type = "Tongue"
        elif path.find("NoMotor") != -1:
            task_type = "NoMotor"
        else:
            print("Cannot find task type in ", path)
        exp_name = path[path.rfind('\\') + 1:]
        subjectid = exp_name[3]
        expid = exp_name[:2]

        self.name = expid + '-' + subjectid + ' ' + \
            str(self.stats['num_classes']) + task_type
        self.classifier_id = str(self.stats['num_classes']) + task_type
        self.filter_id = 'Large'

    def set_stats(self):
        num_classes = []
        hzs = []
        for s in self.stat_list:
            num_classes.append(s['num_classes'])
            hzs.append(s['hz'])
        if not all(c == num_classes[0] for c in num_classes):
            print('ERROR: Number of classes differ!')
        if not all(h == hzs[0] for h in hzs):
            print('ERROR: Hz differ!')

        self.stats = {'num_classes': int(num_classes[0]), 'hz': hzs[0], 'num_channels': 21, 'length': 128}


class task_sampler():
    def __init__(self, task_batch_size, data_batch_size, shuffle=True, train_size=0.80, share=False, train_task_types=None, val_task_ids=None, use_comp=False, use_filters=False):
        all_paths = glob.glob("data/*")
        self.shuffle = shuffle
        self.task_paths = []
        for p in all_paths:
            self.task_paths.append(p)
        self.task_batch_size = task_batch_size
        self.data_batch_size = data_batch_size
        self.train_tasks = []
        self.val_tasks = []
        num_train_tasks = int(len(self.task_paths) * train_size)

        if train_task_types is None and val_task_ids is None:
            tasks = []
            if shuffle:
                self.task_paths = np.random.permutation(self.task_paths)
            for p in self.task_paths:
                if 'Comp' in p:
                    tasks.append(competetask(p, self.data_batch_size))
                elif 'Large' in p:
                    tasks.append(largetask(p, self.data_batch_size))
            self.train_tasks = tasks[:num_train_tasks]
            self.val_tasks = tasks[num_train_tasks:]
        else:
            for p in self.task_paths:
                cut_path = p[p.rfind('\\') + 1:]
                task_id = cut_path[:2]
                task_type = p[p.rfind('-') + 1:]
                if int(task_id) in val_task_ids:
                    print("Val task ", task_id)
                    if 'Comp' in p:
                        if use_comp:
                            self.val_tasks.append(competetask(p, self.data_batch_size))
                    else:
                        self.val_tasks.append(largetask(p, self.data_batch_size))
                elif task_type in train_task_types or train_task_types == "All":
                    print("Train task ", task_id)
                    if 'Comp' in p:
                        if use_comp:
                            self.train_tasks.append(competetask(p, self.data_batch_size))
                    else:
                        self.train_tasks.append(largetask(p, self.data_batch_size))
            if shuffle:
                self.train_tasks = np.random.permutation(
                    self.train_tasks).tolist()
                self.val_tasks = np.random.permutation(self.val_tasks).tolist()

        if share:
            self.classifiers = {}
            if use_filters:
                self.filters = {}
            for ta in self.train_tasks + self.val_tasks:
                self.classifiers[ta.classifier_id] = model.get_classifier(
                    int(ta.stats['num_classes'])).to(device)
                if use_filters:
                    self.filters[ta.filter_id] = model.Filter(
                        ta.stats['num_channels'], ta.stats['length'], 128).to(device)
            for ta in self.train_tasks + self.val_tasks:
                ta.classifier = self.classifiers[ta.classifier_id]
                if use_filters:
                    ta.filter = self.filters[ta.filter_id]
            print(self.classifiers)
            if use_filters:
                print(self.filters)
        self.num_train_iters = int(len(self.train_tasks) / self.task_batch_size)
        self.num_val_iters = int(len(self.val_tasks) / self.task_batch_size)

    def train_iter(self):
        last_i = 0
        for batch in range(self.num_train_iters):
            batch = []
            for i in range(last_i, last_i + self.task_batch_size):
                batch.append(self.train_tasks[i])
                last_i = i
            yield batch
        last_batch = []
        for j in range(last_i, len(self.train_tasks)):
            last_batch.append(self.train_tasks[j])
        if last_batch is not []:
            yield last_batch

        if self.shuffle:
            self.train_tasks = np.random.permutation(self.train_tasks)

    def get_task(self, n):
        return self.train_tasks[n]

    def val_iter(self):
        last_i = 0

        for batch in range(self.num_val_iters):
            batch = []
            for i in range(last_i, last_i + self.task_batch_size):
                batch.append(self.val_tasks[i])
                last_i = i
            yield batch

        last_batch = []
        for j in range(last_i, len(self.val_tasks)):
            last_batch.append(self.val_tasks[j])
        if last_batch is not []:
            yield last_batch

        if self.shuffle:
            self.val_tasks = np.random.permutation(self.val_tasks)
