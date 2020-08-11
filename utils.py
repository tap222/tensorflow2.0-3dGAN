import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import time

# making soem subdirectories, chekcpoint stores the models,
# savepoint saves some created objects at each epoch
def make_directories(checkpoint,savepoint): 
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    if not os.path.exists(savepoint):
        os.makedirs(savepoint)
    
# loads the images and objects into memory 
def make_inputs_and_images(file_batch, voxel_dir):
    voxel_dir+='/'
    models = []
    images = []
    for i,fil in enumerate(file_batch): 
        split = fil.split('/')
        models.append(np.load( voxel_dir+ split[-1].split('_')[0] +'.npy'))
        img = Image.open(fil)
        images.append(np.asarray(img,dtype='uint8'))
    models = np.array(models)
    images = np.array(images)
    start_time = time.time()
    return models, images, start_time
    
def make_inputs(file_batch):
    models = [np.load(f) for f in file_batch]
    models = np.array(models)
    start_time = time.time()
    return models, start_time
    
def save_voxels(save_dir, models, epock, recon_models = None): 
    print("Saving the model")
    np.save(save_dir+str(epock)  , models) 
    if recon_models is not None: 
        np.save(save_dir+str(epock) + '_VAE', recon_models) 

def grab_files_images(image_dir, voxel_dir): 
    files = []
    pattern  = "*.jpg"
    image_dir+='/'
    voxel_dir+='/'
    for dir,_,_ in os.walk(image_dir):
        files.extend(glob(os.path.join(dir,pattern))) 
    voxels = [ v.split('/')[-1].split('.')[0] for v in glob(voxel_dir + '*')]
    
    temp = []
    for f in files: 
        if 'orig_' in f: 
            continue 
        if f.split('/')[-2] not in voxels: continue
        temp.append(f)
    files = []
    valid = [] 
    for i,t in enumerate(temp): 
        if len(valid) < 128 and i %33 == 0 :  
            valid.append(t)
        else:
            files.append(t)
    return files,valid

def grab_files(voxel_dir): 
    voxel_dir+='/'
    return [f for f in glob(voxel_dir + '*.npy')], 0 


def grab_files_surfaces(surface_dir, voxel_dir): 
    surface_dir+='/'
    voxel_dir+='/'
    files = glob(surface_dir+'*')
    voxels = [ v.split('/')[-1].split('_')[-1] for v in glob(voxel_dir + '*')]
    temp = []
    for f in files: 
        if f.split('/')[-1].split('_')[-2] + '.npy' not in voxels: continue
        temp.append(f)
    return temp
    
