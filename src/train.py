from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data


configs = {
    "dataset_root": "/mnt/c/Users/patri/dev/capstone/human-detection/data/testset-jan25",
    "weights_root": "/mnt/c/Users/patri/dev/capstone/human-detection/weights/",
    "pretrained_weights_path": "weights/vgg16_reducedfc.pth",
    "cuda": False,
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "gamma": 0.1,
    "batch_size": 32,
    "weight_decay": 5e-4,
    "num_classes": 2,  # only 2 classes supported at the moment
    "dimension": 300,  # only SSD 300 supported at the moment
    "num_epochs": 10,
}

def setup_directories(configs):
    if not os.path.exists(configs["dataset_root"]):
        os.mkdir(configs["dataset_root"])

    if not os.path.exists(configs["weights_root"]):
        os.mkdir(configs["weights_root"])

def get_pretrained_ssd(configs):
    net = build_ssd('train', configs["dimension"], configs["num_classes"])
    print(f'Loading base network from: {configs["pretrained_weights_path"]}')
    vgg_weights = torch.load(configs["pretrained_weights_path"])
    net.vgg.load_state_dict(vgg_weights)

    if configs["cuda"]:
        net.cuda()

    return net

def get_optimizer(net, configs):
    optimizer = optim.SGD(net.parameters(), 
                          lr=configs["learning_rate"],
                          momentum=configs["momentum"],
                          weight_decay=configs["weight_decay"])
    return optimizer

def get_criterion(configs):
    return MultiBoxLoss(configs["num_classes"], 0.5, True, 0, True, 3, 0.5, False, False)

def get_dataset(configs):
    configs["batch_size"]
    raise NotImplementedError


def train(configs):
    setup_directories(configs)
    net = get_pretrained_ssd(configs)
    optimizer = get_optimizer(net, configs)
    criterion = get_criterion(configs)
    dataset = get_dataset(configs)

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    step_index = 0

    # create batch iterator
    for epoch in range(configs["num_epochs"]):
        for phase in ["train", "val"]:
            if epoch % int(configs["num_epochs"] / 3) == 0 and epoch != 0:
                step_index += 1
                adjust_learning_rate(optimizer, configs["gamma"], step_index)

            # load train data
            images, targets = next(batch_iterator)

            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            print("input images shape: ")
            print(images.shape)
            out = net(images)
            print("Output images shape:")
            print(out[0].shape)
            print(out[1].shape)
            print(out[2].shape)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                        repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               "save_test.pth")

    return net, images


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * gamma ** (step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


if __name__ == '__main__':
    ssd, images = train(configs)
