# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:38:28 2020

@author: dhyakesh
"""

import keras.backend as k
import numpy as np
from keras.layers import (Dense,Input,Activation,Flatten)
from keras.models import Model,Sequential
from keras.layers.convolutional import (Conv2D,MaxPooling2D,AveragePooling2D)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
print (k.common.image_dim_ordering())   
# importing PIL 
from PIL import Image 
import datetime

#utilities function
def BatchNormalization_relu(inputdata):
    bn=BatchNormalization(axis=2)(inputdata)
    activation_relu=Activation(activation="relu")(bn)
    return activation_relu
def BatchNormalization_relu_Conv2d(**arg):
    #get the parameters for conv2d
    filters=arg["filters"]
    strides=arg.setdefault("strides",(1,1))
    padding=arg.setdefault("padding","same")
    kernel_initializer = arg.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = arg.setdefault("kernel_regularizer", l2(1.e-4))
    kernal_size=arg["kernal_size"]
    
    bn_relu=BatchNormalization_relu(arg["inputdata"])
    
    bn_relu_conv2d=Conv2D(
            filters=filters,
            strides=strides,
            padding=padding,
            kernel_size=kernal_size,
            kernel_initializer=kernel_initializer,
            kernel_regularizer= kernel_regularizer
            )(bn_relu)
    return bn_relu_conv2d
        
def _shortcut(inputdata, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    print(inputdata, k.int_shape(inputdata),residual, k.int_shape(residual))
    input_shape = k.int_shape(inputdata)
    residual_shape = k.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
    
    shortcut = inputdata
    """"1 X 1 conv if shape since shape may be different"""
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inputdata)

    return add([shortcut, residual])
  
#create a model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
class resnet(object):
    @staticmethod
    def builder(**builderargs):
        print(builderargs)
        architecture=builderargs.setdefault("architecture",50)
        print(architecture)
        #start building the architechture 
        inputdata=Input(shape=(1000,1000,3))
        conv_1=Conv2D(
            filters=64,
            kernel_size=(7,7),
            strides=(2,2),
            padding="same"
            )(inputdata)
        BN_1=BatchNormalization(axis=2)(conv_1)
        Acti_relu_1=Activation(activation="relu")(BN_1)
        max_pool_1=MaxPooling2D(
              pool_size=(3,3),
              strides=2
                )(Acti_relu_1)
        rinput=max_pool_1
        if(architecture>34):
            print("bottleneck")
            if(architecture==50):
               loops=[3,4,6,3]
               #start filter gets multiplyed by 2 every loop starts
               sfilters=64
               
               for loopindex,repition in enumerate(loops):
                   for iteration in range(repition):
                       print(iteration)
                       if(loopindex!=0 and iteration==0):
                           stridestobeused=2
                       else:
                           stridestobeused=1
                       if(loopindex==0 and iteration==0):
                           #first layer first module
                           
                            conv_1_1 = Conv2D(
                                   filters=sfilters,
                                   kernel_size=1,
                                   strides=(1,1),
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=l2(1.e-4)
                                   )(rinput)
                               
                       else:
                           
                          conv_1_1=BatchNormalization_relu_Conv2d(filters=sfilters,
                                                          strides=stridestobeused,
                                                          kernal_size=1,
                                                          inputdata=rinput)
                       conv_3_3=BatchNormalization_relu_Conv2d(filters=sfilters,
                                                          strides=stridestobeused,
                                                          kernal_size=(3,3),
                                                          inputdata=conv_1_1)
                       residual=BatchNormalization_relu_Conv2d(filters=sfilters*4,
                                                          strides=(1,1),
                                                          kernal_size=(1,1),
                                                          inputdata=conv_3_3)
                       #print(residual)
                       rinput=_shortcut(rinput, residual)
                       
                   sfilters*=2

                       
              
            
        else:
            print("base model")
        block=BatchNormalization_relu(rinput)
        block_shape = k.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block)
        fl=Flatten()(pool2)
        output=Dense(units=2,activation=("softmax"))(fl)
        return(Model(inputs=inputdata,outputs=output))
            
        
        
        
model=resnet.builder(architecture=50)
model.compile(optimizer=Adam(lr=0.01), loss=categorical_crossentropy, metrics=['accuracy'])
model.summary()

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="../real_and_fake_face",target_size=(1000,1000),batch_size=1)
tsdata=ImageDataGenerator()
testdata=tsdata.flow_from_directory(directory="../real_and_fake_face",target_size=(1000,1000),batch_size=1 )
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata ,validation_steps=10,epochs=100)