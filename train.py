import os
import argparse
import re
import glob
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import trainset_loader
from models import hyperfed_LEARN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_block", type=int, default=50)

parser.add_argument("--n_cpu", type=int, default=0)
parser.add_argument("--model_save_path", type=str, default="saved_models/1st")
parser.add_argument('--checkpoint_interval', type=int, default=10)


parser.add_argument("--num_clients", type=int, default=5)
parser.add_argument("--communication", type=int, default=2)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--mode", type=str, default='hyperfed')
parser.add_argument("--mu", type=float, default=1e-6)
opt = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def communication(opt, server_model, models, client_weights):
    with torch.no_grad():
        if opt.mode.lower() == 'hyperfed':
            for key in server_model.state_dict().keys():
                if 'Hyper' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def my_collate(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    prj_data = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, label_data, prj_data, option, feature


def Dataset():
    dataloaders = []
    for i in range(1, 6):
        path = f"dataset/meta_learning/train2/geometry_{i}/train"

        dataset_obj = trainset_loader(path)

        print(f"[DEBUG] Geometry {i} dataset size:", len(dataset_obj))

        dataset = DataLoader(
            dataset_obj,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            collate_fn=my_collate
        )
        dataloaders.append(dataset)
    return dataloaders


class net():
    def __init__(self):
        self.loss = nn.MSELoss()
        self.path = opt.model_save_path
        self.train_datas = Dataset()
        self.start = 0
        self.epoch = opt.epochs
        self.com = opt.communication
        self.client_num = opt.num_clients

        self.server_model = hyperfed_LEARN.Learn(opt.n_block)
        self.server_model.to(device)
        self.models = [copy.deepcopy(self.server_model) for _ in range(self.client_num)]

        self.check_saved_model()
        self.optimizers = [torch.optim.Adam(self.models[idx].parameters(), lr=opt.lr, weight_decay=1e-8) for idx in
                           range(self.client_num)]

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/model_commu_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_str = re.findall(r'model_commu_(-?[0-9]\d*).pth', model)
                    if epoch_str:
                        epoch_num = int(epoch_str[0])
                        if epoch_num > last_epoch:
                            last_epoch = epoch_num
                self.start = last_epoch
                self.server_model.load_state_dict(
                    torch.load('%s/model_commu_%04d.pth' % (self.path, last_epoch), map_location=device))
                for wk_iter in range(self.client_num):
                    self.models[wk_iter].load_state_dict(
                        torch.load('%s/model_worker_id(%04d)_commu_%04d.pth' % (self.path, wk_iter, last_epoch),
                                   map_location=device))

    def initialize_weights(self):
        for module in self.server_model.modules():
            if isinstance(module, hyperfed_LEARN.prj_module):
                nn.init.normal_(module.weight_fed, mean=0.02, std=0.001)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
        print(f"Starting Training on {device} with {self.client_num} clients...")
        for com_iter in range(self.start, self.com):
            for i_wkr in range(self.client_num):
                for epoch in range(self.epoch):
                    for batch_index, data in enumerate(self.train_datas[i_wkr]):
                        input_data, label_data, prj_data, options, feature_vec = data

                        input_data = input_data.to(device)
                        label_data = label_data.to(device)
                        options = options.to(device)
                        feature_vec = feature_vec.to(device)

                        temp = []
                        for i in range(len(prj_data)):
                            temp.append(torch.FloatTensor(prj_data[i]).to(device))
                        prj_data = temp

                        self.optimizers[i_wkr].zero_grad()
                        output = self.models[i_wkr](input_data, prj_data, options, feature_vec)
                        loss = self.loss(output, label_data)
                        loss.backward()
                        self.optimizers[i_wkr].step()

                        print("Com Round: %d | Client: %d | [Epoch %d/%d] [Batch %d/%d]: [loss: %f]" %
                              (com_iter, i_wkr + 1, epoch + 1, self.epoch, batch_index + 1, len(self.train_datas[i_wkr]),
                               loss.item()))

            # Aggregate weights after each communication round
            client_weights = [1 / self.client_num for _ in range(self.client_num)]
            self.server_model, self.models = communication(opt, self.server_model, self.models, client_weights)

            # Save checkpoints
            if opt.checkpoint_interval != -1 and (com_iter + 1) % opt.checkpoint_interval == 0:
                torch.save(self.server_model.state_dict(), '%s/model_commu_%04d.pth' % (self.path, com_iter + 1))
                for check_id in range(self.client_num):
                    torch.save(self.models[check_id].state_dict(),
                               '%s/model_worker_id(%04d)_commu_%04d.pth' % (self.path, check_id, com_iter + 1))


if __name__ == "__main__":
    network = net()
    network.train()

