from model.ssd.layers.modules import MultiBoxLoss
from model.ssd.ssd import build_ssd
import os
import time
from datetime import datetime
import torch
import torch.optim as optim
from utils.visdom import VisdomLinePlotter


configs = {
    "dataset_root": "/mnt/c/Users/patri/dev/capstone/human-detection/data/testset-jan25",
    "weights_root": "/mnt/c/Users/patri/dev/capstone/human-detection/weights/",
    "pretrained_weights_path": "weights/vgg16_reducedfc.pth",
    "cuda": False,
    "initial_lr": 1e-3,
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
                          lr=configs["initial_lr"],
                          momentum=configs["momentum"],
                          weight_decay=configs["weight_decay"])
    return optimizer

def get_criterion(configs):
    return MultiBoxLoss(configs["num_classes"], 0.5, True, 0, True, 3, 0.5, False, False)

def get_dataset(configs):
    configs["batch_size"]
    raise NotImplementedError

def save_weights(net, configs, epoch):
    fn = "ssd_" + str(epoch) + "_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".pth"
    path = os.path.join(configs["weights_root"], fn)
    torch.save(net.state_dict(), path)

def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * gamma ** (step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(configs):
    setup_directories(configs)
    net = get_pretrained_ssd(configs)
    optimizer = get_optimizer(net, configs)
    criterion = get_criterion(configs)
    train_imgs, train_anns, val_imgs, val_anns = get_dataset(configs)
    plotter = VisdomLinePlotter()
    learning_rate = configs["initial_lr"]

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    step_index = 0
    epoch_train_losses = []
    epoch_val_losses = []

    # create batch iterator
    for epoch in range(configs["num_epochs"]):
        for phase in ["train", "val"]:
            if epoch % int(configs["num_epochs"] / 3) == 0 and epoch != 0:
                step_index += 1
                learning_rate, adjust_learning_rate(optimizer, configs["gamma"], step_index, learning_rate)

            imgs, anns = train_imgs, train_anns if phase == "train" else val_imgs, val_anns

            iterations = 0
            losses = []

            t0 = time.time()
            for images, targets in zip(imgs, anns):
                out = net(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()
                losses += [(loc_loss + conf_loss)]

                if iterations % 10 == 0:
                    if phase == "train":
                        plotter.plot(f"Train Loss for Epoch={epoch}", "splitname", f"Train Loss for Epoch={epoch}",
                                     "Epoch", "Loss", range(len(losses)), losses)
                    else:
                        plotter.plot(f"Val Loss for Epoch={epoch}", "splitname", f"Val Loss for Epoch={epoch}", 
                                     "Epoch", "Loss", range(len(losses)), losses)

            t1 = time.time()

            print(f'Epoch {epoch} {phase} time: {(t1 - t0):%.4f} sec.')
            print(f"{phase} Loss: {loss.item():%.4f}")

            # Plot loss
            if phase == "train":
                epoch_train_losses += [(loc_loss + conf_loss) / len(imgs)]
                plotter.plot("Epoch Train Loss", "splitname", "Epoch Train Loss", "Epoch",
                             "Loss", range(len(epoch_train_losses)), epoch_train_losses)
            else:
                epoch_val_losses += [(loc_loss + conf_loss) / len(imgs)]
                plotter.plot("Epoch Val Loss", "splitname", "Epoch Val Loss", "Epoch",
                             "Loss", range(len(epoch_val_losses)), epoch_val_losses)

            if phase == "train":
                save_weights(net, configs, epoch)

    return net, images


if __name__ == '__main__':
    ssd, images = train(configs)
