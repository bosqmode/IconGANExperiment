from Model import GAN
import random, os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import argparse
from DrawBoard import DrawBoard
from tensorflow import summary

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true", help='Add to start training')
parser.add_argument("-l", "--load_weights", action="store_true", help='Add to start drawing')
args = parser.parse_args()

TRAIN = args.train
LOAD_WEIGHTS = args.load_weights
TARGET_PATH = "targets"
EDGE_DIR = "edges"
BATCH_SIZE = 128
G_WEIGHTFILE = "G_Weights"
D_WEIGHTFILE = "D_Weights"
TRAINING_STEPS = 50000
LOGDIR = "logs/log"
IMAGE_SHAPE = (64,64,3)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

GAN = GAN(training = TRAIN, input_shape=IMAGE_SHAPE)

if LOAD_WEIGHTS:
    GAN.Load(G_WEIGHTFILE, D_WEIGHTFILE)

def GetBatch(batch_size = 32):
    real = []
    edges = []
    fakes = []
    for i in range(batch_size):
        choice = random.choice(os.listdir(TARGET_PATH))

        img = cv2.imread(os.path.join(TARGET_PATH,choice))
        img = np.array(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = (img - 127.5) / 127.5

        edge = cv2.imread(os.path.join(EDGE_DIR,choice))
        edge = np.array(edge)
        edge = (edge - 127.5) / 127.5

        fake = GAN.Generator.predict(edge.reshape(-1,IMAGE_SHAPE[0],IMAGE_SHAPE[1],IMAGE_SHAPE[2]))

        edges.append(edge)
        real.append(img)
        fakes.append(fake[0])

    reals = np.array(real)
    edges = np.array(edges)
    fakes = np.array(fakes)

    return edges, reals, fakes

def Train():
    writer = summary.create_file_writer(LOGDIR)
    for e in range(TRAINING_STEPS):
        edges, images, fakes = GetBatch(BATCH_SIZE)

        y_real = np.ones((BATCH_SIZE, 8,8,1))
        d_loss_real = GAN.Discriminator.train_on_batch([edges, images], y_real)
        y_fake = np.zeros((BATCH_SIZE, 8,8,1))
        d_loss_fake = GAN.Discriminator.train_on_batch([edges, fakes], y_fake)
        y = np.ones([BATCH_SIZE, 1])

        g_loss = GAN.GAN.train_on_batch(edges,[y_real,images])
        print("gan_loss_: {0}, d_loss_real: {1}, d_loss_fake: {2}".format(g_loss[0], d_loss_real[0], d_loss_fake[0]))

        if e % 10 == 0:
            edg, img, fakes = GetBatch(1)
            target = img[0]
            target = (target + 1) / 2.0
            edge = edg[0]
            edge = (edge + 1) / 2.0
            fake = fakes[0]
            fake = (fake + 1) / 2.0
            concat = np.vstack([edge,target,fake])
            fake = Image.fromarray(np.uint8(concat*255)).convert('RGB')

            with writer.as_default():
                summary.scalar("g_loss", g_loss[0], step=e)
                summary.scalar("d_loss_real", d_loss_real[0], step=e)
                summary.scalar("d_loss_fake", d_loss_fake[0], step=e)
                summary.image("images", np.array(fake).reshape(-1,fake.height,fake.width,3), step=e)
            writer.flush()
            GAN.Save(G_WEIGHTFILE, D_WEIGHTFILE)

def Draw():
    board = DrawBoard(GAN.Generator)
    while True:
        board.Update()

if TRAIN:
    Train()
else:
    Draw()