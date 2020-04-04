from model.ssd.layers.modules import MultiBoxLoss
from model.ssd.ssd import build_ssd
import os
import time
from datetime import datetime
import torch
import torch.optim as optim
from utils.visdom import VisdomLinePlotter
from utils.image_preprocess import get_dataset
from sklearn.model_selection import train_test_split
import json


configs = {
    "dataset_root": "/home/paperspace/entire_dataset/data/",
    "weights_root": "/home/paperspace/human-detection/saved_weights",
    "pretrained_weights_path": "/home/paperspace/human-detection/pretrained_weights/vgg16_reducedfc.pth",
    "histories_path": "/home/paperspace/human-detection/histories",
    "test_size": 0.2,
    "initial_lr": 1e-3,
    "momentum": 0.9,
    "gamma": 0.1,
    "batch_size": 32,
    "weight_decay": 5e-4,
    "num_classes": 2,  # only 2 classes supported at the moment (including background)
    "dimension": 300,  # only SSD 300 supported at the moment
    "num_epochs": 10,

    "DEBUG": True,
    "debug_batch_size": 1,
    "debug_num_epochs": 100,
    "debug_max_imgs": 10,
    "checkpoint_freq_by_epoch": 80,

    # visdom
    "plot_epoch_losses": False
}


def setup_directories(configs):
    if not os.path.exists(configs["dataset_root"]):
        os.mkdir(configs["dataset_root"])

    if not os.path.exists(configs["weights_root"]):
        os.mkdir(configs["weights_root"])


def get_pretrained_ssd(configs, CUDA):
    net = build_ssd('train', configs["dimension"], configs["num_classes"])
    print(f'Loading base network from: {configs["pretrained_weights_path"]}')
    vgg_weights = torch.load(configs["pretrained_weights_path"])
    net.vgg.load_state_dict(vgg_weights)

    if CUDA:
        net = net.cuda()

    return net


def get_optimizer(net, configs):
    optimizer = optim.SGD(net.parameters(), 
                          lr=configs["initial_lr"],
                          momentum=configs["momentum"],
                          weight_decay=configs["weight_decay"])
    return optimizer


def get_criterion(configs, CUDA):
    return MultiBoxLoss(configs["num_classes"], 0.5, True, 0, True, 3, 0.5, False, CUDA)


def get_train_val_datasets(configs):
    height = width = configs["dimension"]
    images, anns, img_names = get_dataset(configs, height, width, scaling='minmax')
    return train_test_split(images, anns, img_names, test_size=configs["test_size"])


def save_weights(net, configs, epoch):
    fn = "ssd_" + str(epoch) + "_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ".pth"
    path = os.path.join(configs["weights_root"], fn)
    torch.save(net.state_dict(), path)


def adjust_learning_rate(optimizer, gamma, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_history(configs, CUDA, train_img_names, val_img_names):
    configs["CUDA"] = CUDA
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fn = "configs_" + date + ".json"
    pth = os.path.join(configs["histories_path"], fn)
    with open(pth, 'w') as fp:
        json.dump(configs, fp)

    image_history = {
        "train_images": train_img_names,
        "val_images": val_img_names
    }

    fn = "images_" + date + ".json"
    pth = os.path.join(configs["histories_path"], fn)
    with open(pth, 'w') as fp:
        json.dump(image_history, fp)


def train(configs):
    DEBUG = configs["DEBUG"]
    CUDA = torch.cuda.is_available()

    if DEBUG:
        configs["batch_size"] = configs["debug_batch_size"]

    setup_directories(configs)
    net = get_pretrained_ssd(configs, CUDA)
    optimizer = get_optimizer(net, configs)
    criterion = get_criterion(configs, CUDA)
    train_imgs, val_imgs, train_anns, val_anns, train_img_names, val_img_names = get_train_val_datasets(configs)
    plotter = VisdomLinePlotter()
    learning_rate = configs["initial_lr"]

    save_history(configs, CUDA, train_img_names, val_img_names)

    net.train()

    if CUDA:
        print("*****USING CUDA*******")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        net.cuda()

    # loss counters
    epoch_train_losses = []
    epoch_val_losses = []

    if DEBUG:
        configs["num_epochs"] = configs["debug_num_epochs"]
        (train_imgs, train_anns) = (train_imgs[:configs["debug_max_imgs"]], train_anns[:configs["debug_max_imgs"]])
        (val_imgs, val_anns) = (train_imgs, train_anns)

    # create batch iterator
    for epoch in range(configs["num_epochs"]):
        for phase in ["train", "val"]:
            if epoch % max(1, int(configs["num_epochs"] / 3)) == 0 and epoch != 0:
                learning_rate = adjust_learning_rate(optimizer, configs["gamma"], learning_rate)

            (imgs, anns) = (train_imgs, train_anns) if phase == "train" else (val_imgs, val_anns)

            iterations = 0
            losses = []

            t0 = time.time()

            total_loc_loss = 0
            total_conf_loss = 0

            for images, targets in zip(imgs, anns):
                if CUDA:
                    images = images.cuda()
                    targets = [x.cuda() for x in targets]
                
                out = net(images)

                # TEMPORARY FIX
                # if CUDA:
                #     loc_data, conf_data, priors = out
                #     out = loc_data.cuda(), conf_data.cuda(), priors.cuda()

                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                loc_loss = loss_l.item()
                conf_loss = loss_c.item()
                total_loc_loss += loc_loss
                total_conf_loss += conf_loss
                losses += [(loc_loss + conf_loss)]

                if configs["plot_epoch_losses"]:
                    if iterations % 10 == 0:
                        if phase == "train":
                            plotter.plot(f"Train Loss for Epoch={epoch}", "splitname", f"Train Loss for Epoch={epoch}",
                                        "Epoch", "Loss", len(losses) - 1, losses[-1])
                            losses = []
                        else:
                            plotter.plot(f"Val Loss for Epoch={epoch}", "splitname", f"Val Loss for Epoch={epoch}",
                                        "Epoch", "Loss", len(losses) - 1, losses[-1])
                            losses = []

            t1 = time.time()

            print(f'Epoch {epoch} {phase} time: {(t1 - t0):.4f} sec.')
            print(f"{phase} Loss: {(total_loc_loss + total_conf_loss):.4f}")

            # Plot loss
            if phase == "train":
                epoch_train_losses += [(total_loc_loss + total_conf_loss) / len(imgs)]
                plotter.plot("Epoch Train Loss", "splitname", "Epoch Train Loss", "Epoch",
                             "Loss", len(epoch_train_losses) - 1, epoch_train_losses[-1])
            else:
                epoch_val_losses += [(total_loc_loss + total_conf_loss) / len(imgs)]
                plotter.plot("Epoch Val Loss", "splitname", "Epoch Val Loss", "Epoch",
                             "Loss", len(epoch_val_losses) - 1, epoch_val_losses[-1])

            if configs["checkpoint_freq_by_epoch"] % (epoch + 1) == 0 and phase == "train":
                save_weights(net, configs, epoch)

            iterations += 1

    return net, images


if __name__ == '__main__':
    ssd, images = train(configs)
