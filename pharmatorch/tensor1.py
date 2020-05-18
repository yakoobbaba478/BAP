#Importing Libraries

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math
import pandas as pd
import numpy as np
#import torchvision


import fnmatch
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.home import home

import os
import csv
from csv import writer


#function to load the CSV file
def loadCSV(filename):
    print("Entering to load the given csv file")
    with open(filename, 'r') as csvfile:
        df = pd.read_csv(csvfile,  error_bad_lines=False)
        print("loading or reading of file is done")
    csvfile.close()
    return(df)

#function to get the tensor from the given complex
#given -> path of complex folder
def get_voxel(path):
    complex = os.path.join(path)
    for ele in os.listdir(complex):
        if fnmatch.fnmatch(ele, '*_protein.pdb'):
            prot = Molecule(os.path.join(complex, ele))
            prot.filter('protein')

            # If your structure is fully protonated and contains all bond information in prot.bonds skip this step!
            prot = prepareProteinForAtomtyping(prot)


            prot.view(guessBonds=False)
            prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, boxsize=[24, 24, 24], center=[0, 0, 0],
                                                                 buffer=1)
            prot.view(guessBonds=False)
            viewVoxelFeatures(prot_vox, prot_centers, prot_N)

            nchannels = prot_vox.shape[1]

            prot_vox_t = prot_vox.transpose().reshape([1, nchannels, prot_N[0], prot_N[1], prot_N[2]])
            prot_vox_t = torch.tensor(prot_vox_t.astype(np.float32))

    for ele in os.listdir(complex):
        if fnmatch.fnmatch(ele, '*_ligand.mol2'):
            slig = SmallMol(os.path.join(os.path.join(complex, ele)),force_reading=True)
            slig.view(guessBonds=False)

            # For the ligand since it's small we could increase the voxel resolution if we so desire to 0.5 A instead of the default 1 A.
            lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, boxsize=[24, 24, 24], center=[0, 0, 0],
                                                              voxelsize=1, buffer=1)
            slig.view(guessBonds=False)
            viewVoxelFeatures(lig_vox, lig_centers, lig_N)

            lig_vox_t = lig_vox.transpose().reshape([1, nchannels, lig_N[0], lig_N[1], lig_N[2]])
            lig_vox_t = torch.tensor(lig_vox_t.astype(np.float32))

    x = torch.cat((lig_vox_t, prot_vox_t), 1)
    x.squeeze_(0)
    return x


#function to write a tensor in a file
#given -> complex_name and tensor
def writetensor(tensor, complex_name):
    torch.save(tensor, complex_name+".txt")    
    #return file    

#function to get the affinity of the complex 
#given -> complex_name
def get_affinity(complex_name):
    df = loadCSV('/home/bayeslabs/Desktop/yashi/refined-set/out.csv')
    df.fillna(method='ffill', inplace = True)
    for index, row in df.iterrows():
        if row['complex']==complex_name:
            print(row['complex'])
            y = row['affinity']
            break        
    return y        



#function to append input tensor of complex and its affinity details in CSV file
#given -> list containing tesnsorfile and affinity of complex
def appendCSV(listt):
    with open("data.csv", "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(listt)




#current working directly path
dir = os.getcwd()
print(dir)

#refined set directory path
datadir = os.path.join(dir, '/home/bayeslabs/Desktop/yashi/refined-set/refined-set')
print(datadir)

#list of folders inside the refined set
dir_list = os.listdir(datadir)
print(dir_list)
i = list(range(4057))

import multiprocessing
from multiprocessing import Pool


    # for i in range(len(dir_list)):

def f(i):       
    complex_name = dir_list[i]

    path = os.path.join(datadir, dir_list[i])

    # tensor = get_voxel(path)
    #
    # #tensorfile = writetensor(tensor, complex_name)
    # writetensor(tensor, complex_name)

    affinity = get_affinity(complex_name)

    #listt = [complex_name, tensorfile, affinity]
    listt = [complex_name, affinity]

    appendCSV(listt)


def multi_fun():
    with Pool(4) as p:
        p.map(f, i)

multi_fun()






