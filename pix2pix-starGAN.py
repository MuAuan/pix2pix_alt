#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/
#tommyfms2/pix2pix-keras-byt より
#38800
"""
j 8000, Epoch1 8000/8001, Time: 34664.078228235245
10/10 [==============================] - 1s - D logloss: 0.5541 - G tot: 3.1830 - G L1: 0.1752 - G logloss: 1.4312
10/10 [==============================] - 1s - D logloss: 0.5699 - G tot: 3.0943 - G L1: 0.1876 - G logloss: 1.2183
10/10 [==============================] - 1s - D logloss: 0.5579 - G tot: 3.4255 - G L1: 0.2056 - G logloss: 1.3697
10/10 [==============================] - 1s - D logloss: 0.5513 - G tot: 2.9691 - G L1: 0.1629 - G logloss: 1.3404
10/10 [==============================] - 1s - D logloss: 0.5683 - G tot: 2.8544 - G L1: 0.1407 - G logloss: 1.4478
10/10 [==============================] - 1s - D logloss: 0.5625 - G tot: 3.1075 - G L1: 0.1693 - G logloss: 1.4148
patch32
j 8000, Epoch1 8000/8001, Time: 38349.08283615112
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 3.0403 - G L1: 0.1654 - G logloss: 1.3863
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 3.3861 - G L1: 0.2000 - G logloss: 1.3863
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 2.9472 - G L1: 0.1561 - G logloss: 1.3863
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 3.0853 - G L1: 0.1699 - G logloss: 1.3863
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 2.7957 - G L1: 0.1409 - G logloss: 1.3863
10/10 [==============================] - 1s - D logloss: 0.5626 - G tot: 3.0955 - G L1: 0.1709 - G logloss: 1.3863
"""

import os
import argparse

import numpy as np

import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD

import models

def my_normalization(X):
    return X / 127.5 - 1
def my_inverse_normalization(X):
    return (X + 1.) / 2.

#過学習を抑えてしかも少数データで学習するためにｌ２も導入l1_loss⇒l1l2_loss
#encoder-decoder版はl2項を0.5倍する
def l1l2_loss(y_true, y_pred):
    s=K.abs(y_pred - y_true)*(1 + 0.5*K.abs(y_pred - y_true)) 
    return K.sum(s, axis=-1)

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = my_inverse_normalization(X_raw)
    X_proc = my_inverse_normalization(X_proc)
    X_gen = my_inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs,Xg,Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./figures/current_batch_"+suffix+".png")
    plt.clf()
    plt.close()

# tmp load data gray to color
def my_load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_gen"][:].astype(np.float32)
        X_full_train = my_normalization(X_full_train)
        X_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        X_sketch_train = my_normalization(X_sketch_train)
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val
    
# tmp load data gray to color
# for train & test data exactly select each other
def my_load_data_train(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_gen"][:].astype(np.float32)
        X_full_train = my_normalization(X_full_train)
        X_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        X_sketch_train = my_normalization(X_sketch_train)
        return X_full_train, X_sketch_train
    
def my_load_data_test(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_val = hf["val_data_gen"][:].astype(np.float32)
        X_full_val = my_normalization(X_full_val)
        X_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        X_sketch_val = my_normalization(X_sketch_val)
        return X_full_val, X_sketch_val
    
def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc

def training(procImage,rawImage, procImage_val,rawImage_val, X_procImage, X_rawImage, X_procImageIter, X_rawImageIter, j,s, batch_size,patch_size,generator_model,discriminator_model,DCGAN_model):
    b_it = 0
    progbar = generic_utils.Progbar(len(X_procImageIter)*batch_size)
    for (X_proc_batch ,X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
        b_it += 1
        X_disc, y_disc = get_disc_batch(X_proc_batch ,X_raw_batch , generator_model, b_it, patch_size)
        raw_disc, _ = get_disc_batch(X_raw_batch ,X_raw_batch , generator_model, 1, patch_size)
        x_disc = X_disc + raw_disc
        # update the discriminator
        disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

        # create a batch to feed the generator model
        idx = np.random.choice(procImage.shape[0], batch_size)
        X_gen_target, X_gen = procImage[idx], rawImage[idx]
                   
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1

        # Freeze the discriminator
        discriminator_model.trainable = False
        gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
        # or
        # gen_loss =DCGAN_model.fit(X_gen, [X_gen_target, y_gen],batch_size=batch_size,nb_epoch=1,verbose=1,shuffle=True)
        # Unfreeze the discriminator
        discriminator_model.trainable = True

        progbar.add(batch_size, values=[
            ("D logloss", disc_loss),
            ("G tot", gen_loss[0]),
            ("G L1", gen_loss[1]),
            ("G logloss", gen_loss[2])
            ])
        if j % 100==0:
            plot_generated_batch(X_proc_batch ,X_raw_batch ,generator_model, batch_size, "training"+str(s)+"_"+str(j))
            idx = np.random.choice(procImage_val.shape[0], batch_size)
            X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
            plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation"+str(s)+"_"+str(j))
        else:
            continue            

    

def my_train(args):
    # create output finder
    
    if not os.path.exists(os.path.expanduser(args.datasetpath00)):
        os.mkdir(findername)
    
    # create figures
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    
    #ultiN=3
    # load data
    procImage0,rawImage0= my_load_data_train(args.datasetpath00)
    procImage_val0,rawImage_val0 = my_load_data_test(args.datasetpath10)
    procImage1, rawImage1 = my_load_data_train(args.datasetpath01)
    procImage_val1, rawImage_val1 = my_load_data_test(args.datasetpath11)
    procImage2, rawImage2 = my_load_data_train(args.datasetpath02)
    procImage_val2, rawImage_val2 = my_load_data_test(args.datasetpath12)
    procImage3,rawImage3= my_load_data_train(args.datasetpath03)
    procImage_val3,rawImage_val3 = my_load_data_test(args.datasetpath13)
    procImage4, rawImage4 = my_load_data_train(args.datasetpath04)
    procImage_val4, rawImage_val4 = my_load_data_test(args.datasetpath14)
    procImage5, rawImage5 = my_load_data_train(args.datasetpath05)
    procImage_val5, rawImage_val5 = my_load_data_test(args.datasetpath15)
    
    print('procImage.shape : ', procImage0.shape)
    print('rawImage.shape : ', rawImage0.shape)
    print('procImage_val : ', procImage_val0.shape)
    print('rawImage_val : ', rawImage_val0.shape)

    img_shape = rawImage0.shape[-3:]
    print('img_shape : ', img_shape)
    patch_num = (img_shape[0]//args.patch_size) * (img_shape[1] // args.patch_size)
    disc_img_shape = (args.patch_size, args.patch_size, procImage0.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    #generator_model.load_weights('params_generator6_pix_epoch_4800.hdf5')
    # load discriminator model
    discriminator_model = models.my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    #discriminator_model.load_weights('params_discriminator6_pix_epoch_4800.hdf5')
    #loss='mae'
    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.my_load_DCGAN(generator_model, discriminator_model, img_shape, args.patch_size)

    loss = [l1l2_loss, 'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # start training
    
    j=0
    print('start training')
    starttime = time.time()
    
    
    perm = np.random.permutation(rawImage0.shape[0])
    X_procImage0 = procImage0[perm]
    X_rawImage0  = rawImage0[perm]
    X_procImageIter0= [X_procImage0[i:i+args.batch_size] for i in range(0, rawImage0.shape[0], args.batch_size)]
    X_rawImageIter0  = [X_rawImage0[i:i+args.batch_size] for i in range(0, rawImage0.shape[0], args.batch_size)]
    X_procImage1 = procImage1[perm]
    X_rawImage1  = rawImage1[perm]
    X_procImageIter1 = [X_procImage1[i:i+args.batch_size] for i in range(0, rawImage1.shape[0], args.batch_size)]
    X_rawImageIter1  = [X_rawImage1[i:i+args.batch_size] for i in range(0, rawImage1.shape[0], args.batch_size)]
    X_procImage2 = procImage2[perm]
    X_rawImage2  = rawImage2[perm]
    X_procImageIter2 = [X_procImage2[i:i+args.batch_size] for i in range(0, rawImage2.shape[0], args.batch_size)]
    X_rawImageIter2  = [X_rawImage2[i:i+args.batch_size] for i in range(0, rawImage2.shape[0], args.batch_size)]
    X_procImage3 = procImage3[perm]
    X_rawImage3 = rawImage3[perm]
    X_procImageIter3= [X_procImage3[i:i+args.batch_size] for i in range(0, rawImage3.shape[0], args.batch_size)]
    X_rawImageIter3  = [X_rawImage3[i:i+args.batch_size] for i in range(0, rawImage3.shape[0], args.batch_size)]
    X_procImage4 = procImage4[perm]
    X_rawImage4  = rawImage4[perm]
    X_procImageIter4 = [X_procImage4[i:i+args.batch_size] for i in range(0, rawImage4.shape[0], args.batch_size)]
    X_rawImageIter4  = [X_rawImage4[i:i+args.batch_size] for i in range(0, rawImage4.shape[0], args.batch_size)]
    X_procImage5 = procImage5[perm]
    X_rawImage5  = rawImage5[perm]
    X_procImageIter5 = [X_procImage5[i:i+args.batch_size] for i in range(0, rawImage5.shape[0], args.batch_size)]
    X_rawImageIter5  = [X_rawImage5[i:i+args.batch_size] for i in range(0, rawImage5.shape[0], args.batch_size)]
    
    for e in range(args.epoch):
        training(procImage0,rawImage0, procImage_val0,rawImage_val0,X_procImage0,X_rawImage0,X_procImageIter0,X_rawImageIter0,j,0,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        training(procImage1,rawImage1, procImage_val1,rawImage_val1,X_procImage1,X_rawImage1,X_procImageIter1,X_rawImageIter1,j,1,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        training(procImage2,rawImage2, procImage_val2,rawImage_val2,X_procImage2,X_rawImage2,X_procImageIter2,X_rawImageIter2,j,2,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        training(procImage3,rawImage3, procImage_val3,rawImage_val3,X_procImage3,X_rawImage3,X_procImageIter3,X_rawImageIter3,j,3,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        training(procImage4,rawImage4, procImage_val4,rawImage_val4,X_procImage4,X_rawImage4,X_procImageIter4,X_rawImageIter4,j,4,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        training(procImage5,rawImage5, procImage_val5,rawImage_val5,X_procImage5,X_rawImage5,X_procImageIter5,X_rawImageIter5,j,5,args.batch_size,args.patch_size,generator_model,discriminator_model,DCGAN_model)
        
        j += 1
        print("")
        print('j %d, Epoch1 %s/%s, Time: %s' % (j, e + 1, args.epoch, time.time() - starttime))
        if j % 100==0:
            generator_model.save_weights('params_generator6_pix_epoch_{0:03d}.hdf5'.format(j), True)
            discriminator_model.save_weights('params_discriminator6_pix_epoch_{0:03d}.hdf5'.format(j), True)
        else:
            continue        
        


def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    parser.add_argument('--datasetpath00', '-d_train0', type=str, default ="sugao2egaoRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath01', '-d_train1', type=str, default ="egao2ikariRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath02', '-d_train2', type=str, default ="ikari2kuyasiRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath03', '-d_train3', type=str, default ="kuyasi2nakiRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath04', '-d_train4', type=str, default ="naki2henRaw_train.hdf5")  #,  "required=True)
    parser.add_argument('--datasetpath05', '-d_train5', type=str, default ="hen2sugaoRaw_train.hdf5")  #,  "required=True)
 
    parser.add_argument('--datasetpath10', '-d_test0', type=str, default ="sugao2egaoRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath11', '-d_test1', type=str, default ="egao2ikariRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath12', '-d_test2', type=str, default ="egao2ikariRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath13', '-d_test3', type=str, default ="kuyasi2nakiRaw_test.hdf5")   #, required=True)
    parser.add_argument('--datasetpath14', '-d_test4', type=str, default ="sugao2mayu.hdf5")   #, required=True)
    parser.add_argument('--datasetpath15', '-d_test5', type=str, default ="mayu2sugao.hdf5")   #, required=True)

    parser.add_argument('--patch_size', '-p', type=int, default=64)
    parser.add_argument('--batch_size', '-b', type=int, default=5)
    parser.add_argument('--epoch','-e', type=int, default=8001)
    args = parser.parse_args()

    K.set_image_data_format("channels_last")

    my_train(args)


if __name__=='__main__':
    main()


