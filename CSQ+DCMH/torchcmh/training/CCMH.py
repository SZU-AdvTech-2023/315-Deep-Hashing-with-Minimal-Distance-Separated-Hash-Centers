import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD,RMSprop
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models import cnnf, MLP
from torchcmh.models.CSQ.Net import  ResNet, MoCo ##Binary_hash,
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.dataset import single_data
from torchcmh.loss.CSQLoss import OurLoss
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CCMH(TrainBase):

    def __init__(self, data_name, img_dir, bit, visdom=False, batch_size=128, cuda=True, **kwargs):
        super(CCMH, self).__init__("CCMH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.data_name = data_name
        self.loss_store = ["log loss", 'quantization loss', 'balance loss', 'loss','center_loss','pair_loss', 'CSQloss']
        self.parameters = {'gamma': 1, 'eta': 1}
        self.max_epoch = 500
        self.lr = {'img': 0.1, 'txt': 0.1}
        self.lr_decay_freq = 10
        self.lr_decay = (1e-6 / 10**(-1.5))**(1/self.max_epoch)
        # self.lr = {'img': 0.1, 'txt': 0.1}
        # self.lr_decay_freq = 10
        # self.lr_decay = 0.5
        self.num_train = len(self.train_data)

        self.txt_model = MLP.MLP(self.train_data.get_tag_length(), bit, leakRelu=False)
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        self.train_L = self.train_data.get_all_label()
        self.ones = torch.ones(batch_size, 1)
        self.ones_ = torch.ones(self.num_train - batch_size, 1)
        if cuda:

            self.txt_model = self.txt_model.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
            self.ones = self.ones.cuda()
            self.ones_ = self.ones_.cuda()
        self.Sim = calc_neighbor(self.train_L, self.train_L)
        self.B = torch.sign(self.F_buffer + self.G_buffer)
        kwargs = {
            "optimizer": {
                "type": RMSprop,
                # "epoch_lr_decrease": 30,
                "optim_param": {
                    "lr": 1e-5,
                    "weight_decay": 1e-5,
                    # "momentum": 0.9
                    # "betas": (0.9, 0.999)
                },
            },
            "label_size": 100,
            "net": MoCo,
            "mome": 0.9,
            "n_gpu": torch.cuda.device_count(),
            "without_BN": False,
        }
        self.img_model = kwargs['net'](kwargs, bit, kwargs['label_size'])
        if cuda:
            self.img_model = kwargs['net'](kwargs, bit, kwargs['label_size']).cuda()

        optimizer_img = kwargs['optimizer']['type'](self.img_model.parameters(), **(kwargs['optimizer']['optim_param']))
        #optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'])
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'])
        self.optimizers = [optimizer_img, optimizer_txt]

        self._init()

    def train(self, bit, **kwargs):
        # 这里开始改
        num_works = 4

        kwargs = {
            "label_size": 100,
            "mome": 0.9,
            "n_class": 24,
            "epoch_change": 9,
            "net": MoCo,
            'remarks' : "Ours",
            "beta": 1.0,
            "lambda": 0.0001,
            "max_norm": 5.0,
        }
        flag = True
        script_dir = os.path.dirname(__file__)
        full_path = os.path.join(script_dir, f"../models/CSQ/centerswithoutVar/CSQ_init_{flag}_{kwargs['n_class']}_{bit}.npy")
        full_path = os.path.normpath(full_path)
        kwargs["center_path"]=  full_path
        full_path = os.path.join(script_dir, f"../models/CSQ/results/{self.data_name}/{kwargs['remarks']}/ours_{bit}.npy")
        full_path = os.path.normpath(full_path)
        kwargs["save_center"] = full_path
        kwargs['num_train'] = len(self.train_data)
        l = list(range(kwargs['n_class']))
        self.criterion = OurLoss(kwargs, bit, l)
        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 1:
            self.img_model = torch.nn.DataParallel(self.img_model)
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        # 这里结束改
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works, shuffle=False, pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.img_load()
            self.train_data.re_random_item()
            self.img_model.train()

            train_loss = 0
            train_center_loss = 0
            train_pair_loss = 0

            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor

                image = data['img']  # type: torch.Tensor
                self.optimizers[0].zero_grad()
                if self.cuda:
                    image = image.cuda()
                    sample_L = sample_L.cuda()

                u1, u2 = self.img_model(image, None, None)
                self.F_buffer[ind, :] = u1.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)
                logloss, quantization, balance = self.object_function(u1, sample_L, G, F, ind)
                CSQloss, center_loss, pair_loss = self.criterion(u1, u2, sample_L, ind, epoch)

                CSQloss = torch.as_tensor(CSQloss, dtype=torch.float32)
                center_loss = torch.as_tensor(center_loss, dtype=torch.float32)
                pair_loss = torch.as_tensor(pair_loss, dtype=torch.float32)

                loss = CSQloss + logloss + self.parameters['gamma'] * quantization + self.parameters['eta'] * balance
                loss /= (self.num_train * self.batch_size)
                #print(CSQloss)
                if self.n_gpu:
                    CSQloss = loss.mean()
                    center_loss = center_loss.mean()
                    pair_loss = pair_loss.mean()
                train_loss += loss.item()
                train_center_loss += center_loss.item()
                train_pair_loss += pair_loss.item()


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.img_model.parameters(), 5.0)
                self.optimizers[0].step()


                self.loss_store['log loss'].update(logloss.item(), (self.batch_size * self.num_train))
                self.loss_store['quantization loss'].update(quantization.item(), (self.batch_size * self.num_train))
                self.loss_store['center_loss'].update(balance.item(), (self.batch_size * self.num_train))
                self.loss_store['pair_loss'].update(balance.item(), (self.batch_size * self.num_train))
                self.loss_store['CSQloss'].update(CSQloss.item())
                self.loss_store['loss'].update(loss.item())

            self.print_loss(epoch)
            self.plot_loss("img loss")
            self.reset_loss()

            self.train_data.txt_load()
            self.train_data.re_random_item()

            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                text = data['txt']  # type: torch.Tensor
                if self.cuda:
                    text = text.cuda()
                    sample_L = sample_L.cuda()

                cur_g = self.txt_model(text)  # cur_g: (batch_size, bit)
                self.G_buffer[ind, :] = cur_g.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                # calculate loss
                logloss, quantization, balance = self.object_function(cur_g, sample_L, F, G, ind)
                loss = logloss + self.parameters['gamma'] * quantization + self.parameters['eta'] * balance
                loss /= (self.num_train * self.batch_size)

                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()

                self.loss_store['log loss'].update(logloss.item(), (self.batch_size * self.num_train))
                self.loss_store['quantization loss'].update(quantization.item(), (self.batch_size * self.num_train))
                self.loss_store['balance loss'].update(balance.item(), (self.batch_size * self.num_train))
                self.loss_store['loss'].update(loss.item())
            #
            #(self.F_buffer.shape)
            #print(self.G_buffer.shape)
            self.print_loss(epoch)
            self.plot_loss('text loss')



            self.reset_loss()
            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.valid(epoch)
            self.lr_schedule()
            # self.plotter.next_epoch()
        print("train finish")

    def object_function(self, cur_h: torch.Tensor, sample_label: torch.Tensor, A: torch.Tensor, C: torch.Tensor,ind):
        unupdated_ind = np.setdiff1d(range(self.num_train), ind)
        S = calc_neighbor(sample_label, self.train_L)
        theta = 1.0 / 2 * torch.matmul(cur_h, A.t())
        logloss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
        quantization = torch.sum(torch.pow(self.B[ind, :] - cur_h, 2))
        balance = torch.sum(torch.pow(cur_h.t().mm(self.ones) + C[unupdated_ind].t().mm(self.ones_), 2))
        return logloss, quantization, balance


def train(dataset_name: str, img_dir: str, bit: int, visdom=False, batch_size=128, cuda=True, **kwargs):
    trainer = CCMH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train(bit, **kwargs)


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss
