import numpy as np
import random
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Activation, Convolution2D, MaxPooling2D, Lambda, Input
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.api.keras.optimizers import SGD
from tensorflow.contrib.keras.api.keras import backend as K
from setup_mnist import MNIST
from setup_codec import CODEC
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
import h5py
from numpy import load
from keras.utils.np_utils import to_categorical
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class CatDog:
    def __init__(self):
        # load and confirm the shape
        data = load('dogs_vs_cats_photos.npy')
        labels = load('dogs_vs_cats_labels.npy')
        catg_labels = to_categorical(labels, num_classes=2)
        print(data.shape, labels.shape)
        
        VALIDATION_SIZE = 2500
        
        self.validation_data = data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = labels[:VALIDATION_SIZE]
        self.validation_catg_labels = catg_labels[:VALIDATION_SIZE]
        self.train_data = data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = labels[VALIDATION_SIZE:]
        self.train_catg_labels = catg_labels[VALIDATION_SIZE:]


def train_autoencoder(codec, batch_size=1000, epochs=1000, saveFilePrefix=None, use_tanh=True, train_imagenet=False, aux_num=0):

    """Train autoencoder

    python3 train_CAE.py -d mnist --compress_mode 1 --epochs 10000 --save_prefix mnist

    """

    ckptFileName = saveFilePrefix + "ckpt"

    encoder_model_filename = saveFilePrefix + "encoder.json"
    decoder_model_filename = saveFilePrefix + "decoder.json"
    encoder_weight_filename = saveFilePrefix + "encoder.h5"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    data = CatDog()

    if os.path.exists(decoder_weight_filename):
        print("Load the pre-trained model.")
        codec.decoder.load_weights(decoder_weight_filename)
    elif os.path.exists(ckptFileName):
        print("Load the previous checkpoint")
        codec.decoder.load_weights(ckptFileName)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)
    codec.decoder.compile(loss='mse', optimizer=sgd)

    checkpointer = ModelCheckpoint(filepath=ckptFileName, verbose=1, save_best_only=True)
    if train_imagenet:
        # train using Keras imageGenerator
        print("Training using imageGenerator:")
        codec.decoder.fit_generator(
                    data.train_generator_flow,
                    steps_per_epoch=100,
                    epochs=epochs,
                    validation_data=data.validation_generator_flow,
                    validation_steps=100,
                    callbacks = [checkpointer])
    else: 
        # in-memory training
        x_train = data.train_data
        # add zeros for correction
        if aux_num > 0:
            x_train = np.concatenate((x_train, np.zeros((aux_num,)+ x_train.shape[1:])), axis=0)
            x_train = np.concatenate((x_train, 0.5*np.ones((aux_num,)+ x_train.shape[1:])), axis=0)
            x_train = np.concatenate((x_train, -0.5*np.ones((aux_num,)+ x_train.shape[1:])), axis=0)

        y_train = x_train
        x_test = data.validation_data
        y_test = x_test
        print("In-memory training:")
        print("Shape of training data:{}".format(x_train.shape))
        print("Shape of testing data:{}".format(x_test.shape))

        
        codec.decoder.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              epochs=epochs,
              shuffle=True,
              callbacks = [checkpointer])

    print("Checkpoint is saved to {}\n".format(ckptFileName))


    
    model_json = codec.encoder.to_json()
    with open(encoder_model_filename, "w") as json_file:
        json_file.write(model_json)
    print("Encoder specification is saved to {}".format(encoder_model_filename))

    codec.encoder.save_weights(encoder_weight_filename)
    print("Encoder weight is saved to {}\n".format(encoder_weight_filename))

    model_json = codec.decoder.to_json()
    with open(decoder_model_filename, "w") as json_file:
        json_file.write(model_json)

    print("Decoder specification is saved to {}".format(decoder_model_filename))

    codec.decoder.save_weights(decoder_weight_filename)
    print("Decoder weight is saved to {}\n".format(decoder_weight_filename))

def main(args):
    # load data
    print("Start training autoencoder")
    codec = CODEC(img_size=64, num_channels=3, 
                compress_mode=args["compress_mode"], resize=64)
    train_autoencoder(codec,  batch_size=args["batch_size"], 
            epochs=args["epochs"], saveFilePrefix=args["save_prefix"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_prefix", default="codec", help="prefix of file name to save trained model/weights under model folder")
    parser.add_argument("--compress_mode", type=int, choices=[1, 2, 3], default=1, help="the compress mode, 1:1/4 2:1/16, 3:1/64")
    parser.add_argument("--batch_size", default=64, type=int, help="the batch size when training autoencoder")
    parser.add_argument("--epochs", default=60, type=int, help="the number of training epochs")
    parser.add_argument("--seed", type=int, default=9487)
    parser.add_argument("--imagenet_train_dir", default=None, help="The path to training data for imagenet")
    parser.add_argument("--imagenet_validation_dir", default=None, help="The path to validation data for imagenet")
    parser.add_argument("--aux_num", default=0, type=float, help="The number of data for calibration")
    parser.add_argument("--train_data_source", help="the training data other than the default dataset, MNIST only")

    
    args = vars(parser.parse_args())
    if not os.path.isdir("codec"):
        print("Folder for saving models does not exist. The folder is created.")
        os.makedirs("codec")
    args["save_prefix"] = "codec/" + args["save_prefix"] + "_" + str(args["compress_mode"]) + "_"

    # setup random seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    print(args)



    main(args)