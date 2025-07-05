#!/usr/bin/env python
# coding: utf-8

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt, cos, sin, floor
from data_utils import *
import tensorflow as tf
import pickle 

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"{len(gpus)} GPUs found")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("no GPUs found")


from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow..v1.keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error
from mango.tuner import Tuner
from scipy.stats import uniform
from keras_flops import get_flops
import pickle
import csv
import random
import itertools
import quaternion
import math
from hardware_utils import *
import time


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r


# Check if GPU is available and being used by TensorFlow
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA")
else:
    print("TensorFlow is NOT built with CUDA")

# Print device placement for operations
print("\nDevice placements:")
with tf.device('/CPU:0'):
    a = tf.random.normal([1000, 1000])
    print("CPU operation:", tf.reduce_sum(a))

try:
    with tf.device('/GPU:0'):
        b = tf.random.normal([1000, 1000])
        print("GPU operation:", tf.reduce_sum(b))
except RuntimeError as e:
    print("GPU operation failed:", str(e))


# ## Import Training, Validation and Test Set

sampling_rate = 200
window_size = 400
stride = 20
data_folder = 'dataset_download_and_splits/' #dataset directory

fn_val = data_folder + 'validation_test_data.pkl'
fn_train = data_folder + 'training_data.pkl'

if not os.path.exists(fn_val):
    #Training Set
    X, Y_disp, Y_head, Y_pos, x0_list, y0_list, size_of_each, x_vel, y_vel, z_vel, head_s, head_c, X_orig, Physics_Vec= import_aqualoc_dataset(type_flag = 2,
                                usePhysics=True, dataset_folder = data_folder, useMagnetometer = True, returnOrientation = False, 
                emulateDVL=False, AugmentationCopies = 0, sampling_rate = sampling_rate, resampling_rate = 0, window_size = window_size, stride = stride, verbose=False)
    P = np.repeat(Physics_Vec,window_size).reshape((Physics_Vec.shape[0],window_size,1))
    X = np.concatenate((X,P),axis=2)

    #Validation Set
    X_val, Y_disp_val, Y_head_val, Y_pos_val, x0_list_val, y0_list_val, size_of_each_val, x_vel_val, y_vel_val, z_vel_val, head_s_val, head_c_val, X_orig_val, Physics_Vec_val= import_aqualoc_dataset(type_flag = 3,
                                usePhysics=True, dataset_folder = data_folder, useMagnetometer = True, returnOrientation = False, 
                emulateDVL=False, AugmentationCopies = 0, sampling_rate = sampling_rate, resampling_rate = 0, window_size = window_size, stride = stride, verbose=False)
    #Test Set
    X_test, Y_disp_test, Y_head_test, Y_pos_test, x0_list_test, y0_list_test, size_of_each_test, x_vel_test, y_vel_test, z_vel_test, head_s_test, head_c_test, X_orig_test, Physics_Vec_test = import_aqualoc_dataset(type_flag = 4,
                                usePhysics=True, dataset_folder = data_folder, useMagnetometer = True, returnOrientation = False, 
                emulateDVL=False, AugmentationCopies = 0, sampling_rate = sampling_rate, resampling_rate = 0, window_size = window_size, stride = stride, verbose=False)
    P_test = np.repeat(Physics_Vec_test,window_size).reshape((Physics_Vec_test.shape[0],window_size,1))
    X_test = np.concatenate((X_test,P_test),axis=2)


    # In[8]:
    # Save training data to pickle file
    training_data = {
        'X': X,
        'Y_disp': Y_disp,
        'Y_head': Y_head, 
        'Y_pos': Y_pos,
        'x0_list': x0_list,
        'y0_list': y0_list,
        'size_of_each': size_of_each,
        'x_vel': x_vel,
        'y_vel': y_vel,
        'z_vel': z_vel,
        'head_s': head_s,
        'head_c': head_c,
        'X_orig': X_orig,
        'Physics_Vec': Physics_Vec
    }

    import pickle
    with open(fn_train, 'wb') as data_folder:
        pickle.dump(training_data, data_folder)


    # Save validation and test data to pickle file
    validation_data = {
        'X_val': X_val,
        'Y_disp_val': Y_disp_val, 
        'Y_head_val': Y_head_val,
        'Y_pos_val': Y_pos_val,
        'x0_list_val': x0_list_val,
        'y0_list_val': y0_list_val,
        'size_of_each_val': size_of_each_val,
        'x_vel_val': x_vel_val,
        'y_vel_val': y_vel_val, 
        'z_vel_val': z_vel_val,
        'head_s_val': head_s_val,
        'head_c_val': head_c_val,
        'X_orig_val': X_orig_val,
        'Physics_Vec_val': Physics_Vec_val,
        'X_test': X_test,
        'Y_disp_test': Y_disp_test,
        'Y_head_test': Y_head_test, 
        'Y_pos_test': Y_pos_test,
        'x0_list_test': x0_list_test,
        'y0_list_test': y0_list_test,
        'size_of_each_test': size_of_each_test,
        'x_vel_test': x_vel_test,
        'y_vel_test': y_vel_test,
        'z_vel_test': z_vel_test,
        'head_s_test': head_s_test,
        'head_c_test': head_c_test,
        'X_orig_test': X_orig_test,
        'Physics_Vec_test': Physics_Vec_test
    }

    with open(fn_val, 'wb') as data_folder:
        pickle.dump(validation_data, data_folder)

    P_val= np.repeat(Physics_Vec_val,window_size).reshape((Physics_Vec_val.shape[0],window_size,1))
    X_val = np.concatenate((X_val,P_val),axis=2)
else:
    # Load training data from pickle file
    with open(fn_train, 'rb') as f:
        training_data = pickle.load(f)

    # Unpack training data variables
    X = training_data['X']
    Y_disp = training_data['Y_disp']
    Y_head = training_data['Y_head']
    Y_pos = training_data['Y_pos']
    x0_list = training_data['x0_list']
    y0_list = training_data['y0_list']
    size_of_each = training_data['size_of_each']
    x_vel = training_data['x_vel']
    y_vel = training_data['y_vel']
    z_vel = training_data['z_vel']
    head_s = training_data['head_s']
    head_c = training_data['head_c']
    X_orig = training_data['X_orig']
    Physics_Vec = training_data['Physics_Vec']

    # Load validation and test data from pickle file
    with open(fn_val, 'rb') as f:
        validation_data = pickle.load(f)

    # Unpack validation data variables
    X_val = validation_data['X_val']
    Y_disp_val = validation_data['Y_disp_val']
    Y_head_val = validation_data['Y_head_val']
    Y_pos_val = validation_data['Y_pos_val']
    x0_list_val = validation_data['x0_list_val']
    y0_list_val = validation_data['y0_list_val']
    size_of_each_val = validation_data['size_of_each_val']
    x_vel_val = validation_data['x_vel_val']
    y_vel_val = validation_data['y_vel_val']
    z_vel_val = validation_data['z_vel_val']
    head_s_val = validation_data['head_s_val']
    head_c_val = validation_data['head_c_val']
    X_orig_val = validation_data['X_orig_val']
    Physics_Vec_val = validation_data['Physics_Vec_val']

    P_val= np.repeat(Physics_Vec_val,window_size).reshape((Physics_Vec_val.shape[0],window_size,1))
    X_val = np.concatenate((X_val,P_val),axis=2)

    # Unpack test data variables
    X_test = validation_data['X_test']
    Y_disp_test = validation_data['Y_disp_test']
    Y_head_test = validation_data['Y_head_test']
    Y_pos_test = validation_data['Y_pos_test']
    x0_list_test = validation_data['x0_list_test']
    y0_list_test = validation_data['y0_list_test']
    size_of_each_test = validation_data['size_of_each_test']
    x_vel_test = validation_data['x_vel_test']
    y_vel_test = validation_data['y_vel_test']
    z_vel_test = validation_data['z_vel_test']
    head_s_test = validation_data['head_s_test']
    head_c_test = validation_data['head_c_test']
    X_orig_test = validation_data['X_orig_test']
    Physics_Vec_test = validation_data['Physics_Vec_test']

from tensorflow.keras import layers, Model, Input

def build_model(input_shape, nb_filters:int = 32, kernel_size:int = 7, dilations:list = [1, 2, 4, 8, 16, 32, 64, 128], 
                dropout_rate:float = 0.0, use_skip_connections:bool = False, norm_flag:bool=False):
    inputs = Input(shape=input_shape)

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout_rate=dropout_rate,
        use_skip_connections=use_skip_connections,
        use_batch_norm=(norm_flag)
    )(inputs)

    x = layers.Reshape((nb_filters, 1))(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='linear', name='pre')(x)

    outputs = [
        layers.Dense(1, activation='linear', name='velx')(x),
        layers.Dense(1, activation='linear', name='vely')(x)
    ]

    return Model(inputs=inputs, outputs=outputs)

def objective_NN(epochs=500,nb_filters=32,kernel_size=7,dilations=[1, 2, 4, 8, 16, 32, 64, 128],
                dropout_rate=0,
                use_skip_connections=False,norm_flag=0, batch_size=512):

    inval = 0
    rmse_vel_x = 'inf'
    rmse_vel_y = 'inf'
    timesteps = window_size
    input_dim =  X.shape[2] # + 1

    model = build_model(input_shape=(timesteps, input_dim),
                        nb_filters=nb_filters,
                        kernel_size=kernel_size,
                        dilations=list(dilations),
                        dropout_rate=dropout_rate,
                        use_skip_connections=use_skip_connections,
                        norm_flag=bool(norm_flag))
    opt = tf.keras.optimizers.Adam()
    # opt = tf.keras.optimizers.legacy.Adam()
    model.compile(loss={'velx': 'mse','vely':'mse'}, optimizer=opt, run_eagerly=False)

    Flops = get_flops(model, batch_size=1)
    convert_to_tflite_model(model=model,training_data=X,quantization=quantization,output_name=output_name) 
    maxRAM, maxFlash = return_hardware_specs(device)

    if(HIL==True):
        convert_to_cpp_model(dirpath)
        RAM, Flash, Latency, idealArenaSize, errorCode = HIL_controller(dirpath=dirpath,
                                                                       chosen_device=device,
                                                                       window_size=window_size, 
                                                                        number_of_channels = input_dim,
                                                                       quantization=quantization)     
        score = -5.0
        if(Flash==-1):
            row_write = [score, rmse_vel_x,rmse_vel_y,RAM,Flash,Flops,Latency,
                 nb_filters,kernel_size,dilations,dropout_rate,use_skip_connections,norm_flag]
            print('Design choice:',row_write)
            with open(log_file_name, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row_write)
            return score

        elif(Flash!=-1):
            checkpoint = ModelCheckpoint(model_name, monitor='loss',
                                         verbose=1,
                                         save_best_only=True,                         
                                        mode='auto',
                                        save_freq='epoch')
            model.fit(x=X, y=[x_vel, y_vel],epochs=epochs, shuffle=True,callbacks=[checkpoint],batch_size=batch_size)     
            model = load_model(model_name,custom_objects={'TCN': TCN})
            y_pred = model.predict(X_val)
            rmse_vel_x = mean_squared_error(x_vel_val, y_pred[0], squared=False)
            rmse_vel_y = mean_squared_error(y_vel_val, y_pred[1], squared=False)
            model_acc = -(rmse_vel_x+rmse_vel_y) 
            resource_usage = (RAM/maxRAM) + (Flash/maxFlash) 
            score = model_acc + 0.01*resource_usage - 0.05*Latency  #weigh each component as you like

            row_write = [score, rmse_vel_x,rmse_vel_y,RAM,Flash,Flops,Latency,
                 nb_filters,kernel_size,dilations,dropout_rate,use_skip_connections,norm_flag]
            print('Design choice:',row_write)
            with open(log_file_name, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(row_write)

    else:
        score = -5.0
        Flash = os.path.getsize(output_name)
        RAM = get_model_memory_usage(batch_size=1,model=model)
        Latency=-1
        max_flops = (30e6)

        if(RAM < maxRAM and Flash<maxFlash):
            checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
            model.fit(x=X, y=[x_vel, y_vel],epochs=epochs, shuffle=True,callbacks=[checkpoint],batch_size=batch_size)     
            model = load_model(model_name,custom_objects={'TCN': TCN})
            y_pred = model.predict(X_val)
            rmse_vel_x = mean_squared_error(x_vel_val, y_pred[0], squared=False)
            rmse_vel_y = mean_squared_error(y_vel_val, y_pred[1], squared=False)
            model_acc = -(rmse_vel_x+rmse_vel_y) 
            resource_usage = (RAM/maxRAM) + (Flash/maxFlash)
            score = model_acc + 0.01*resource_usage - 0.05*(Flops/max_flops)  #weigh each component as you like

        row_write = [score, rmse_vel_x,rmse_vel_y,RAM,Flash,Flops,Latency,
                 nb_filters,kernel_size,dilations,dropout_rate,use_skip_connections,norm_flag]
        print('Design choice:',row_write)
        with open(log_file_name, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row_write)  

    return score

def objfunc(args_list):

    objective_evaluated = []

    start_time = time.time()

    for hyper_par in args_list:
        nb_filters = hyper_par['nb_filters']
        kernel_size = hyper_par['kernel_size']
        dropout_rate = hyper_par['dropout_rate']
        use_skip_connections = hyper_par['use_skip_connections']
        norm_flag=hyper_par['norm_flag']
        dil_list = hyper_par['dil_list']

        objective = objective_NN(epochs=model_epochs,nb_filters=nb_filters,kernel_size=kernel_size,
                                 dilations=dil_list,
                                 dropout_rate=dropout_rate,use_skip_connections=use_skip_connections,
                                 norm_flag=norm_flag)
        objective_evaluated.append(objective)

        end_time = time.time()
        print('objective:', objective, ' time:',end_time-start_time)

    return objective_evaluated

def save_res(data, file_name):
    pickle.dump( data, open( file_name, "wb" ) )

# ## Training and NAS


device = "NUCLEO_F746ZG" #hardware name
model_name = 'TD_Aqualoc_'+device+'.keras'
dirpath="./bin/" #hardware program directory
HIL = False  #use real hardware or proxy?
quantization = False #use quantization or not?
model_epochs = 20 #epochs to train each model for
NAS_epochs = 30 #epochs for hyperparameter tuning
output_name = 'g_model.tflite'
log_file_name = 'log_NAS_Aqualoc_'+device+'.csv'
fn_tuning_results = log_file_name[0:-4]+'.p'

if not os.path.exists(log_file_name):
    # os.remove(log_file_name)

    row_write = ['score', 'rmse_vel_x','rmse_vel_y','RAM','Flash','Flops','Latency',
                     'nb_filters','kernel_size','dilations','dropout_rate','use_skip_connections','norm_flag']
    with open(log_file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row_write)
    if os.path.exists(fn_tuning_results):
        os.remove(fn_tuning_results)

    min_layer = 3
    max_layer = 8
    a_list = [1,2,4,8,16,32,64,128,256]
    all_combinations = []
    dil_list = []
    for r in range(len(a_list) + 1):
        combinations_object = itertools.combinations(a_list, r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    all_combinations = all_combinations[1:]
    for item in all_combinations:
        if(len(item) >= min_layer and len(item) <= max_layer):
            dil_list.append((item))

    # import pdb; pdb.set_trace()
    param_dict = {
        'nb_filters': range(2,64),
        'kernel_size': range(2,16),
        'dropout_rate': np.arange(0.0,0.5,0.1),
        'use_skip_connections': [True, False],
        'norm_flag': np.arange(0,1),
        'dil_list': dil_list
    }

    conf_Dict = dict()
    conf_Dict['batch_size'] = 1
    conf_Dict['num_iteration'] = NAS_epochs
    conf_Dict['initial_random']= 5
    tuner = Tuner(param_dict, objfunc, conf_Dict)

    print("Tune")
    all_runs = []
    results = tuner.maximize()
    all_runs.append(results)
    save_res(all_runs, fn_tuning_results)
else:
    with open(fn_tuning_results, 'br') as fh:
        all_runs = pickle.load(fh)
    results = all_runs[-1]
         
print(" ## Train the Best Model")

batch_size = 512
model_epochs = 300
nb_filters = results['best_params']['nb_filters']
kernel_size = results['best_params']['kernel_size']
dilations = results['best_params']['dil_list']
dropout_rate = results['best_params']['dropout_rate']
use_skip_connections = results['best_params']['use_skip_connections']
norm_flag = results['best_params']['norm_flag']

input_dim =  X.shape[2]
model = build_model(input_shape=(window_size, input_dim),
					nb_filters=nb_filters,
					kernel_size=kernel_size,
					dilations=list(dilations),
					dropout_rate=dropout_rate,
					use_skip_connections=use_skip_connections,
					norm_flag=bool(norm_flag))

# opt = tf.keras.optimizers.legacy.Adam()
opt = tf.keras.optimizers.Adam()
model.compile(loss={'velx': ['mse', pearson_r],'vely': ['mse', pearson_r]},optimizer=opt)  

checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True)
model.fit(x=X, y=[x_vel, y_vel],epochs=model_epochs, shuffle=True,callbacks=[checkpoint],
        batch_size=batch_size) 


# ## Evaluate the Best Model

# you're a physics engineer. When return code, embed it in triple backtick blocks.

# #### Velocity Prediction RMSE

model = load_model(model_name,custom_objects={'TCN': TCN})
y_pred = model.predict(X_test)
rmse_vel_x = mean_squared_error(x_vel_test, y_pred[0], squared=False)
rmse_vel_y = mean_squared_error(y_vel_test, y_pred[1], squared=False)
print('Vel_X RMSE, Vel_Y RMSE:',rmse_vel_x,rmse_vel_y)


# #### ATE and RTE Metrics

a = 0
b = size_of_each_test[0]
ATE = []
RTE = []
ATE_dist = []
RTE_dist = []
for i in range(len(size_of_each_test)):
    X_test_sel = X_test[a:b,:,:]
    x_vel_test_sel = x_vel_test[a:b]
    y_vel_test_sel = y_vel_test[a:b]
    Y_head_test_sel = Y_head_test[a:b]
    Y_disp_test_sel = Y_disp_test[a:b]
    if(i!=len(size_of_each_test)-1):
        a += size_of_each_test[i]
        b += size_of_each_test[i]

    y_pred = model.predict(X_test_sel)

    pointx = []
    pointy = []
    Lx =  x0_list_test[i]
    Ly = y0_list_test[i]
    for j in range(len(x_vel_test_sel)):
        Lx = Lx + (x_vel_test_sel[j]/(((window_size-stride)/stride)))
        Ly = Ly + (y_vel_test_sel[j]/(((window_size-stride)/stride)))    
        pointx.append(Lx)
        pointy.append(Ly)   
    Gvx = pointx
    Gvy = pointy

    pointx = []
    pointy = []
    Lx =  x0_list_test[i]
    Ly = y0_list_test[i]
    for j in range(len(x_vel_test_sel)):
        Lx = Lx + (y_pred[0][j]/(((window_size-stride)/stride)))
        Ly = Ly + (y_pred[1][j]/(((window_size-stride)/stride)))
        pointx.append(Lx)
        pointy.append(Ly)
    Pvx = pointx
    Pvy = pointy    

    at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=sampling_rate,window_size=window_size,stride=stride)
    ATE.append(at)
    RTE.append(rt)
    ATE_dist.append(Cal_len_meters(Gvx, Gvy))
    RTE_dist.append(Cal_len_meters(Gvx, Gvy, 600))
    print('ATE, RTE, Trajectory Length, Trajectory Length (60 seconds)',ATE[i],RTE[i],ATE_dist[i],RTE_dist[i])

print('Median ATE and RTE', np.median(ATE),np.median(RTE))


# #### Sample Trajectory Plotting

# In[ ]:


#you can use the size_of_each_test variable to control the region to plot. We plot for the first trajectory
a = 0
b = size_of_each_test[0]

X_test_sel = X_test[a:b,:,:]
x_vel_test_sel = x_vel_test[a:b]
y_vel_test_sel = y_vel_test[a:b]
Y_head_test_sel = Y_head_test[a:b]
Y_disp_test_sel = Y_disp_test[a:b]

y_pred = model.predict(X_test_sel)

pointx = []
pointy = []
Lx =  x0_list_test[i]
Ly = y0_list_test[i]
for j in range(len(x_vel_test_sel)):
    Lx = Lx + (x_vel_test_sel[j]/(((window_size-stride)/stride)))
    Ly = Ly + (y_vel_test_sel[j]/(((window_size-stride)/stride)))    
    pointx.append(Lx)
    pointy.append(Ly)   
Gvx = pointx
Gvy = pointy

pointx = []
pointy = []
Lx =  x0_list_test[i]
Ly = y0_list_test[i]
for j in range(len(x_vel_test_sel)):
    Lx = Lx + (y_pred[0][j]/(((window_size-stride)/stride)))
    Ly = Ly + (y_pred[1][j]/(((window_size-stride)/stride)))
    pointx.append(Lx)
    pointy.append(Ly)
Pvx = pointx
Pvy = pointy  

print('Plotting Trajectory of length (meters): ',Cal_len_meters(Gvx, Gvy))

ptox = Pvx
ptoy = Pvy

plt.plot(Gvx,Gvy,label='Ground Truth',color='salmon')
plt.plot(ptox,ptoy,label='TinyOdom',color='green',linestyle='-')
plt.grid()
plt.legend(loc='best')
plt.title('UUV - Aqualoc Dataset')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.show()


# #### Error Evolution

#For the last trajectory

a = 0
b = size_of_each_test[0]

X_test_sel = X_test[a:b,:,:]
x_vel_test_sel = x_vel_test[a:b]
y_vel_test_sel = y_vel_test[a:b]
Y_head_test_sel = Y_head_test[a:b]
Y_disp_test_sel = Y_disp_test[a:b]

y_pred = model.predict(X_test_sel)

pointx = []
pointy = []
Lx =  x0_list_test[i]
Ly = y0_list_test[i]
for j in range(len(x_vel_test_sel)):
    Lx = Lx + (x_vel_test_sel[j]/(((window_size-stride)/stride)))
    Ly = Ly + (y_vel_test_sel[j]/(((window_size-stride)/stride)))    
    pointx.append(Lx)
    pointy.append(Ly)   
Gvx = pointx
Gvy = pointy

pointx = []
pointy = []
Lx =  x0_list_test[i]
Ly = y0_list_test[i]
for j in range(len(x_vel_test_sel)):
    Lx = Lx + (y_pred[0][j]/(((window_size-stride)/stride)))
    Ly = Ly + (y_pred[1][j]/(((window_size-stride)/stride)))
    pointx.append(Lx)
    pointy.append(Ly)
Pvx = pointx
Pvy = pointy  

at, rt, at_all, rt_all = Cal_TE(Gvx, Gvy, Pvx, Pvy,
                                    sampling_rate=sampling_rate,window_size=window_size,stride=stride)

x_ax = np.linspace(0,60,len(rt_all))
print('Plotting for trajectory of length (meters): ',Cal_len_meters(Gvx, Gvy))

plt.plot(x_ax,rt_all,label='TinyOdom',color='green',linestyle='-')
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Position Error (m)')
plt.title('UUV - Aqualoc Dataset')
plt.grid()
plt.show()


# ## Deployment

# #### Conversion to TFLite

# In[ ]:


convert_to_tflite_model(model=model,training_data=X_tr,quantization=quantization,output_name='g_model.tflite') 


# #### Conversion to C++

# In[ ]:


convert_to_cpp_model(dirpath)

