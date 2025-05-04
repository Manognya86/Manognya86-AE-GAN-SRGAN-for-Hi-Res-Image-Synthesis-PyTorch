import os
import sys
import tarfile
import math
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
tf = tf.compat.v1
tf.disable_eager_execution()
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from scipy import linalg, misc

from database import Session, ModelEvaluation
from inference_api import AEGANInference

# ======================
# CONFIG
# ======================
MODEL_DIR = "models/latest"
MODEL_VERSION = os.path.basename(os.readlink(MODEL_DIR)) if os.path.islink(MODEL_DIR) else "unknown"
INCEPTION_MODEL_DIR = "/tmp/imagenet"
INCEPTION_MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
SOFTMAX = None

cfg = tf.ConfigProto()
cfg.gpu_options.allow_growth = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ======================
# INCEPTION INITIALIZATION
# ======================
def _init_inception():
    global SOFTMAX
    if not os.path.exists(INCEPTION_MODEL_DIR):
        os.makedirs(INCEPTION_MODEL_DIR)

    filename = INCEPTION_MODEL_URL.split('/')[-1]
    filepath = os.path.join(INCEPTION_MODEL_DIR, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(INCEPTION_MODEL_URL, filepath, _progress)
        print('\nDownloaded model:', filename)

    tarfile.open(filepath, 'r:gz').extractall(INCEPTION_MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(INCEPTION_MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session(config=cfg) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        pool3_flat = tf.reshape(pool3, [-1, 2048])
        logits = tf.matmul(pool3_flat, w)
        SOFTMAX = tf.nn.softmax(logits)

# ======================
# PSNR & SSIM
# ======================
def calculate_psnr(original, enhanced):
    mse = torch.mean((original - enhanced) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_ssim(original, enhanced):
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    mu_x, mu_y = torch.mean(original), torch.mean(enhanced)
    sigma_x, sigma_y = torch.std(original), torch.std(enhanced)
    sigma_xy = torch.mean((original - mu_x) * (enhanced - mu_y))
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))

def run_psnr_ssim(test_dataset_path):
    print("🔍 Evaluating PSNR & SSIM...")
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    dataset = dset.ImageFolder(root=test_dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    total_psnr = total_ssim = count = 0
    enhancer = AEGANInference(MODEL_DIR)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(enhancer.device)
            enhanced = enhancer.aegan_model(images)
            total_psnr += calculate_psnr(images, enhanced)
            total_ssim += calculate_ssim(images, enhanced)
            count += 1

    avg_psnr, avg_ssim = total_psnr / count, total_ssim / count

    with Session() as session:
        session.add(ModelEvaluation(model_version=MODEL_VERSION, psnr=avg_psnr, ssim=avg_ssim, evaluated_at=datetime.utcnow()))
        session.commit()

    print(f"✅ PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

# ======================
# INCEPTION SCORE
# ======================
def get_images(path, image_size):
    files = os.listdir(path)
    random.shuffle(files)
    images = []
    for file in files:
        img = Image.open(os.path.join(path, file)).convert('RGB').resize((image_size, image_size))
        images.append(np.array(img))
    return images

def get_inception_score(images, log_file, splits=10):
    assert(type(images) == list and type(images[0]) == np.ndarray)
    inps = [np.expand_dims(img.astype(np.float32), 0) for img in images]
    bs = 100

    with tf.Session(config=cfg) as sess:
        preds = []
        for img in images:
            img = np.expand_dims(img, axis=0)  # shape becomes (1, 64, 64, 3)
            pred = sess.run(SOFTMAX, {'ExpandDims:0': img})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        log_filename = os.path.basename(log_file).replace('.', '_') + ".txt"
        os.makedirs("record", exist_ok=True)
        with open(os.path.join("record", log_filename), "w") as file:
            scores = []
            for i in range(splits):
                part = preds[i * preds.shape[0] // splits:(i + 1) * preds.shape[0] // splits]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                score = np.exp(np.mean(np.sum(kl, 1)))
                scores.append(score)
                file.write(f"score of split {i}: {score}\n")
            file.write(f"mean score: {np.mean(scores)}\n")
            file.write(f"std score: {np.std(scores)}\n")

        return np.mean(scores), np.std(scores)

# ======================
# FID CALCULATION
# ======================
def calculate_fid_given_paths(paths, inception_path):
    def read_images(path):
        all_images = []

        def crop_center(img, cropx, cropy):
            y, x, _ = img.shape
            startx = x // 2 - cropx // 2
            starty = y // 2 - cropy // 2
            return img[starty:starty+cropy, startx:startx+cropx]

        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Single image grid (like real_samples.png)
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            tile_size = 64  # adjust based on your grid
            h, w, _ = img.shape
            rows, cols = h // tile_size, w // tile_size
            for i in range(rows):
                for j in range(cols):
                    crop = img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                    resized = np.array(Image.fromarray(crop).resize((512, 512), Image.BILINEAR))
                    all_images.append(crop_center(resized, 512, 512))
        elif os.path.isdir(path):
            # Folder of .png images
            for file in sorted(os.listdir(path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = np.array(Image.open(os.path.join(path, file)).convert("RGB"))
                    img = np.array(Image.fromarray(img).resize((512, 512), Image.BILINEAR))
                    all_images.append(crop_center(img, 512, 512))
        else:
            raise ValueError(f"Invalid path: {path}")
        return np.array(all_images)


    def calculate_stats(images):
        act = get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def get_activations(images):
        with tf.Session(config=cfg) as sess:
            layer = sess.graph.get_tensor_by_name('pool_3:0')
            acts = []
            for img in images:
                img_exp = np.expand_dims(img, axis=0)  # shape: (1, 512, 512, 3)
                pred = sess.run(layer, {'ExpandDims:0': img_exp})  # shape: (1, 1, 1, 2048)
                acts.append(pred.reshape(1, -1))  # shape: (1, 2048)
            return np.concatenate(acts, axis=0)  # final shape: (N, 2048)


    def frechet_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def crop_center(img, cropx, cropy):
        y, x, _ = img.shape
        return img[y//2 - cropy//2:y//2 + cropy//2, x//2 - cropx//2:x//2 + cropx//2]

    create_inception_graph(inception_path)
    imgs1 = read_images(paths[0])
    imgs2 = read_images(paths[1])
    mu1, sigma1 = calculate_stats(imgs1)
    mu2, sigma2 = calculate_stats(imgs2)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def create_inception_graph(pth):
    with tf.gfile.FastGFile(pth + '/classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')



def plot_metrics(log_file='metrics_log.csv'):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['psnr'], label='PSNR', marker='o')
    plt.plot(df['epoch'], df['ssim'], label='SSIM', marker='x')
    plt.plot(df['epoch'], df['fid'], label='FID', marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Model Evaluation Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("record/metric_plot.png")
    plt.show()

def show_comparison(real, fake, num=5):
    fig, axs = plt.subplots(num, 2, figsize=(6, 2 * num))
    for i in range(num):
        axs[i, 0].imshow(real[i])
        axs[i, 0].set_title("Real")
        axs[i, 1].imshow(fake[i])
        axs[i, 1].set_title("Generated")
        for j in range(2):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("record/comparison.png")
    plt.show()


# ======================
# MAIN
# ======================
if __name__ == '__main__':
    import urllib.request

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--real', required=True)
    parser.add_argument('--generated', required=True)
    parser.add_argument('--log_name', required=True)
    parser.add_argument('--imageSize', type=int, default=64)
    args = parser.parse_args()

    if SOFTMAX is None:
        _init_inception()

    print(f"🚀 Evaluating model: {args.dataset}")
    run_psnr_ssim(args.dataroot)
    fid_score = calculate_fid_given_paths([args.real, args.generated], INCEPTION_MODEL_DIR)
    print(f"✅ FID Score: {fid_score:.2f}")
    run_inception_score = get_inception_score(get_images(args.generated, args.imageSize), args.log_name)
    print(f"✅ Inception Score: Mean={run_inception_score[0]:.2f}, Std={run_inception_score[1]:.2f}")
