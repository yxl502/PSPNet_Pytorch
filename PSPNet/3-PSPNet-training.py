from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import pandas as pd

root_path = './data/VOCdevkit/VOC2012/'

train_img_list, train_anno_list, val_img_list, val_anno_list = \
    make_datapath_list(rootpath=root_path)

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list,
                           train_anno_list, phase='train',
                           transform=DataTransform(
                               input_size=475, color_mean=color_mean,
                               color_std=color_std
                           ))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                         transform=DataTransform(
                             input_size=475, color_mean=color_mean,
                             color_std=color_std
                         ))

batch_size = 8
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

from utils.pspnet import PSPNet

net = PSPNet(n_classes=150)

state_dict = torch.load('./weights/pspnet50_ADE20K.pth')

net.load_state_dict(state_dict)

n_classes = 21

net.decode_feature.classification = nn.Conv2d(
    in_channels=512, out_channels=n_classes,
    kernel_size=1, stride=1, padding=0
)

net.aux.classification = nn.Conv2d(
    in_channels=256, out_channels=n_classes,
    kernel_size=1, stride=1, padding=0
)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


net.decode_feature.classification.apply(weights_init)
net.aux.classification.apply(weights_init)

print('网络设置完毕：成功载入事先训练完毕的权重。')


# 设置损失函数


class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight * loss_aux

criterion = PSPLoss(aux_weight=0.4)

# 利用调读器调整每轮epoch的学习率
optimizer = optim.SGD([
    {'params': net.feature_conv.parameters(), 'lr': 1e-3},
    {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
    {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
    {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
    {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
    {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params': net.decode_feature.parameters(), 'lr': 1e-2},
    {'params': net.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.0001)


def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1 - epoch / max_epoch), 0.9)


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)


def train_model(net, dataloaders_dict, criterion,
                scheduler, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用的设备: ', device)

    net.to(device)
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)

    batch_size = dataloaders_dict['train'].batch_size

    iteration = 1
    logs = []

    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('============================')
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('============================')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                scheduler.step()
                optimizer.zero_grad()
                print('(train)')
            else:
                if (epoch + 1) % 5 == 0:
                    net.eval()
                    print('(val)')
                else:
                    continue

            count = 0
            for images, anno_class_images in dataloaders_dict[phase]:
                if images.size()[0] == 1:
                    continue

                images = images.to(device)
                anno_class_images = anno_class_images.to(device)

                if phase == 'train' and count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    loss = criterion(
                        outputs, anno_class_images.long()
                    ) / batch_multiplier

                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start

                            print('迭代 {} || Loss: {:.4f} || 10 iter: {:.4f} sec.'.format(
                                iteration, loss.item() / batch_size * batch_multiplier,
                                duration
                            ))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

            t_epoch_finish = time.time()

            print('----------------')
            print('Epoch {} || Epoch_Train_Loss: {:.4f} || Epoch_Val_Loss: {:.4f}'.format(
                epoch + 1, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs
            ))

            print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

            t_epoch_start = time.time()

            log_epoch = {'epoch': epoch + 1, 'train_loss': epoch_train_loss / num_train_imgs,
                         'val_loss': epoch_val_loss / num_val_imgs}

            logs.append(log_epoch)

            df = pd.DataFrame(logs)
            df.to_csv('log_output.csv')

        torch.save(net.state_dict(), 'weigths/pspnet50_' + str(epoch + 1) + '.pth')


num_epochs = 30
train_model(net, dataloaders_dict, criterion, scheduler,
            optimizer, num_epochs=num_epochs)
