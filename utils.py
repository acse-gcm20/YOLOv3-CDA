import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from pytorchyolo.test import evaluate_model_file
from sklearn.metrics import auc
from PIL import Image

# Utility function to write lists to a text file
def list_to_file(list_array, filename):
    dims = len(np.shape(list_array))
    
    if dims == 1:
        rows = len(list_array)
        cols = 1
    else:
        rows = len(list_array[0])
        cols = len(list_array)

    with open(filename, 'w') as f:
        for row in range(rows):
            vals = [list_array[col][row] for col in range(cols)]
            f.write(("{} "*cols+'\n').format(*vals))

def plot_loss(stats_file, save_path=None):
    training_losses = []
    validation_losses = []

    with open(stats_file) as f:
        for line in f:
            training_losses.append(float(line.rstrip("\n").split()[0]))
            validation_losses.append(float(line.rstrip("\n").split()[1]))

    epochs = len(training_losses)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

    ax.plot(range(1, epochs+1), training_losses, label="Training Loss")
    ax.plot(range(1, epochs+1), validation_losses, '--', label="Validation Loss")
    ax.set_xlim(1, epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

# Statistics utility functions
def plot_stats(stats_file, save_path=None):
    training_losses = []
    validation_losses = []
    precision_vals = []
    recall_vals = []
    mAP_vals = []

    with open(stats_file) as f:
        for line in f:
            tl, vl, precision, recall = line.rstrip("\n").split()[0:4]
            training_losses.append(float(tl))
            validation_losses.append(float(vl))
            precision_vals.append(float(precision))
            recall_vals.append(float(recall))

            if len(line.rstrip("\n").split()) == 5:
                mAP_vals.append(float(line.rstrip("\n").split()[4]))

    epochs = len(training_losses)

    fig, axs = plt.subplots(2, 2, figsize=(7, 8))

    axs[0,0].plot(range(1, epochs+1), training_losses, label="Training Loss")
    axs[0,0].plot(range(1, epochs+1), validation_losses, '--', label="Validation Loss")
    axs[0,0].set_xlim(1, epochs)
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].legend()
    axs[0,0].grid(True)

    axs[0,1].plot(range(1, epochs+1), mAP_vals)
    axs[0,1].set_xlim(1, epochs)
    axs[0,1].set_ylim(-0.1, 1.1)
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("mAP")
    axs[0,1].grid(True)

    axs[1,0].scatter(range(1, epochs+1), recall_vals)
    axs[1,0].set_xlim(1, epochs)
    axs[1,0].set_ylim(-0.1, 1.1)
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("Recall")
    axs[1,0].grid(True)

    axs[1,1].scatter(range(1, epochs+1), precision_vals)
    axs[1,1].set_xlim(1, epochs)
    axs[1,1].set_ylim(-0.1, 1.1)
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("Precision")
    axs[1,1].grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

# Class to calculate, save and plot Precision-Recall curves
class PRcurve:
    def __init__(self, weights_path, start=0.05, stop=0.5, n=10):
        self.confs = np.linspace(start, stop, num=n)
        self.precisions = []
        self.recalls = []

        for i, conf in enumerate(self.confs):
            print("\n%i/%i: Confidence Threshold = %.3f" % (i+1, n, conf))
            metrics = evaluate_model_file('yolov3.cfg', weights_path, 'data/valid.txt', ['crater'], batch_size=32, n_cpu=2, conf_thres=conf, verbose=False)
            self.precisions.append(float(np.mean(metrics[0])))
            self.recalls.append(float(np.mean(metrics[1])))
            print("Precision = %.3f, Recall = %.3f" % (np.mean(metrics[0]), np.mean(metrics[1])))

        list_to_file([self.precisions, self.recalls], "PRstats.txt")
        self.auc_score = auc(self.recalls, self.precisions)

    def save_stats(self, filename):
        list_to_file([self.precisions, self.recalls], filename)

    def plot(self, save_path=None):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        ax.plot(self.recalls, self.precisions)
        ax.set_title("Validation Set Precision-Recall Curve, AUC = %.2f" % self.auc_score)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')

def get_rects(fname, label_dir, img_scale=1, center=True):
    """Calcualte the rects from bounding box labels"""
    rects = []
    pth = f'{label_dir}/{fname.rstrip(".png")}.txt'

    if center:
        shift = 1
    else:
        shift = 0

    if os.path.getsize(pth) != 0:
        with open(pth, 'r') as f:
            labels = f.readlines()
        
        labels = [label.rstrip('\n') for label in labels]

        for label in labels:
            label = label.split(' ')
            state = label[0]
            w = float(label[3])*img_scale
            h = float(label[4])*img_scale
            x = float(label[1])*img_scale - (w/2 * shift)
            y = float(label[2])*img_scale - (h/2 * shift)
            rects.append([state, x, y, w, h])   

    return rects

def plot_image_dir(image_dir, label_dir, label=True, save_path=None):
    """Plot a random sample of images from a directory with bounding boxes"""
    imgs = os.listdir(image_dir)

    fig = plt.figure(figsize=(7,12))
    max_i = len(os.listdir(image_dir))
    for i in range(6):
        r = random.randint(0, max_i-1)
        fname = imgs[r]
        im = Image.open(f'{image_dir}/{fname}')

        # Create figure and axes
        ax = fig.add_subplot(3,2,i+1)

        # Display the image
        ax.imshow(im)
        ax.set_title(fname)

        # Create a Rectangle patch
        rects = get_rects(fname, label_dir, img_scale=593)
        for coords in rects:
            rect = patches.Rectangle((coords[1], coords[2]), coords[3], coords[4], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if label:
                ax.text(coords[1], coords[2]-10, coords[0])

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

def comparison_plot(img_source, label_source, detections_dir, num, save_path=None):
    """Plot comparisons of ground truth and detected labels"""
    detection_files = os.listdir(detections_dir)
    fnames = [name.rstrip('.txt') for name in detection_files]

    images = [f'{img_source}/{name}.png' for name in fnames]
    label_files = [f'{label_source}/{name}.txt' for name in fnames]
    detections = [f'{detections_dir}/{name}.txt' for name in fnames]

    fig, axs = plt.subplots(num, 2, figsize=(7, 4*num))

    #for i, (img, label, detection) in enumerate(zip(images, label_files, detections)):
    for i in range(num):
        r = random.randint(0, len(os.listdir(img_source))-1)
        img = images[r]
        label = label_files[r]
        detection = detections[r]
        
        fname = os.path.basename(img).rstrip('.png')
        im = Image.open(img)
        ax = axs[i]

        # Ground Truth
        ax[0].imshow(im)
        ax[0].set_title(f'{fname} Ground Truth')
        
        # Add labels
        rects = get_rects(fname, label_source, img_scale=593)
        for coords in rects:
            rect = patches.Rectangle((coords[1], coords[2]), coords[3], coords[4], linewidth=2, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
            if label:
                ax[0].text(coords[1], coords[2]-8, coords[0], color='r', fontsize=12, fontweight='bold')

        # Detections
        ax[1].imshow(im)
        ax[1].set_title(f'{fname} Detections')

        # Add labels
        rects = get_rects(fname, detections_dir, center=False)
        for coords in rects:
            rect = patches.Rectangle((coords[1], coords[2]), coords[3], coords[4], linewidth=2, edgecolor='b', facecolor='none')
            ax[1].add_patch(rect)
            if label:
                ax[1].text(coords[1], coords[2]-8, coords[0], color='b', fontsize=12, fontweight='bold')

        # if i == num-1:
        #     break

    plt.show()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

def list_file_2_dir(list_file, dir):
    """Read list of images and move images and corresponding labels to desired dir"""
    os.makedirs(dir+'/images/', exist_ok=True)
    os.makedirs(dir+'/labels/', exist_ok=True)

    with open(list_file) as f:
        imgs = f.readlines()

    imgs = [img.rstrip(' \n') for img in imgs]
    for img_path in imgs:
        (img_dir, img) = os.path.split(img_path)
        lbl_dir = img_dir.rstrip('images')+'labels'
        lbl = img.rstrip('png')+'txt'

        os.system(f'cp {img_dir}/{img} data/test_set/images')
        os.system(f'cp {lbl_dir}/{lbl} data/test_set/labels')

    print(f'{len(os.listdir("data/test_set/images/"))} images')
    print(f'{len(os.listdir("data/test_set/labels/"))} labels')