import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import vision_transformer as vits
import PIL
import argparse
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import tqdm


def get_dataset(args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    dataset = datasets.ImageFolder(args.data_path, transform=transform)

    return dataset


def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * \
                np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, model_type):
    num_cls_tokens = 2 if "distilled" in model_type else 1

    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # Sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # Now average across all the tokens

    return mean_distances


def attention(model, data_loader, device, args):
    count = 0
    attns_all = np.zeros((12, 12, 197, 197))
    for image, _ in tqdm.tqdm(data_loader):
        image = image.to(device)
        attns = model(image, return_type="attn")
        attns = np.array([x.cpu().numpy() for x in attns])
        attns = np.mean(attns, axis=1)
        attns_all += attns

        count += 1

    attns_all /= count
    filename = args.model_path[:-4]
    np.save(f"{filename}_attn.npy", attns_all)

    attention = attns_all
    # Get the number of heads from the mean distance output.
    num_heads = attention.shape[1]
    num_layer = attention.shape[0]

    mean_distances = {}
    for layer in range(num_layer):
        mean_distances[f"{layer}"] = compute_mean_attention_dist(patch_size=16,
                                                                 attention_weights=np.expand_dims(
                                                                     attention[layer], axis=(0)),
                                                                 model_type='norm')
    plt.cla()
    plt.figure(figsize=(9, 9))
    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        plt.scatter(x=x, y=y)

    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Attention Distance", fontsize=14)
    plt.title(f"{os.path.basename(args.model_path)[:-4]}", fontsize=14)
    plt.xticks(range(0, 13))
    plt.yticks(range(0, 160, 20))

    plt.grid()
    plt.savefig(
        f"{args.model_path[:-4]}_attn.jpg", bbox_inches='tight')


def svd(model, data_loader, device, args):
    svd = np.zeros((12, 5))
    count = 0
    for img, _ in tqdm.tqdm(data_loader):
        img = img.to(device)
        count += img.shape[0]
        target = model(img, return_type="svd")

        target = [x.cpu().numpy() for x in target]
        target = np.array(target)
        target = target / np.linalg.norm(target, axis=-1, keepdims=True)

        # svd on single instance
        for id_index in range(target.shape[1]):
            for layer_index in range(target.shape[0]):
                s, v, d = np.linalg.svd(target[layer_index, id_index])
                svd[layer_index][0] += v[:1].sum() / v.sum()
                svd[layer_index][1] += v[:2].sum() / v.sum()
                svd[layer_index][2] += v[:3].sum() / v.sum()
                svd[layer_index][3] += v[:4].sum() / v.sum()
                svd[layer_index][4] += v[:5].sum() / v.sum()

    svd = svd / count
    np.save(f"{args.model_path[:-4]}_svd.npy", svd)

    plt.cla()
    x = np.arange(12)
    for index_ in range(5):
        plt.plot(x, svd[:, index_], label=str(index_+1))

    plt.legend(loc="upper right")
    plt.xticks(range(0, 13), fontsize=20)
    plt.yticks([0.07, 0.17, 0.27, 0.37, 0.47, 0.57, 0.67], fontsize=17)
    plt.grid(linestyle="--")
    plt.savefig(
        f"{args.model_path[:-4]}_svd.jpg", bbox_inches='tight')


def main(args):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = vits.__dict__['vit_base'](num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    dataset = get_dataset(args)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=6,
                                              pin_memory=True,
                                              batch_size=args.batch_size, drop_last=False)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)  # the missing keys should be [].

    attention(model, data_loader, device, args)
    svd(model, data_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="~/84.5_dbot_base_pre.pth", help="pre-trained model path")
    parser.add_argument("--data_path", type=str,
                        default="~/datasets/Imagenet2012/vaL", help="imagenet val dataset")
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    main(args)
