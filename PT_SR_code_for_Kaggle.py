""" Pulse train and Shift Register implementation on GPU with CUDA
"""
import numpy as np
import cupy as cp
from datetime import datetime
import math
from sys import exit
from time import sleep
from random import random

bins0 = cp.arange(-0.5,256,1)
mul1 = cp.arange(256)

def generate_pt0(dw,fr,dm,T): #generate pulse-train, no dead-time effect
    pt0 = cp.array([0])
    N_min = 10**12 # minimum number of neutrons in all multiplicities (see below)
    
    for i in range(dm.shape[0]):
        M = i + 1 # current multiplciity

        if dm[i]>0:
            R = fr*dm[i] # rate of current multiplicity
            sc = 1/R # scale for cp.random.exponential
            N = int(T*R) # number of event in time interval T
            if N_min>N:
                N_min=N
            arr1 = cp.random.exponential(sc,N)
            arr1 = cp.cumsum(arr1)
            arr1 = cp.tile(arr1,M)
            arr1 += cp.random.exponential(dw,N*M)
            pt0 = cp.append(pt0,arr1)
    
    pt0.sort()
    n_start = 200000 # change back to 4000 # number of first neutrons to remove, fixed to 4000 to speed-up
    n_stop = int(6*len(pt0)/N_min**0.5) # number of last neutron to remove, 6 sigma
    pt0 = pt0[n_start:-n_stop]
    del arr1
    return pt0

def apply_deadtime(pt0,dt):
    pt1 = cp.copy(pt0)
    if dt > 0:
        arr1 = cp.diff(pt0)
        pt1 = pt1[1:]
        mask = arr1 >= dt
        pt1 = pt1[mask]
        del arr1, mask
    return pt1

def calculate_raa(pt1,P,G,dw):
    LG = 25*dw # long gate delay
    n_stop = cp.searchsorted(pt1,cp.array(pt1[-1]-30*dw-10*G))
    
    # RA gate calcs
    G_time = cp.copy(pt1[10:n_stop])    
    G_time += P # RA windows start times, P us pre-delay      
    G_ind = cp.searchsorted(pt1,G_time)
    G_time += G
    RA_num = cp.searchsorted(pt1,G_time)-G_ind
    
    # A gate calcs
    G_time += LG # A window start time after long delay
    G_ind = cp.searchsorted(pt1,G_time)
    G_time += G
    A_num = cp.searchsorted(pt1,G_time)-G_ind
    
    RA_h, junk = cp.histogram(RA_num,bins=bins0)
    A_h, junk = cp.histogram(A_num,bins=bins0)
    Tc = G_time[-1]-G_time[0] #cycle time (us)
    
    del RA_num, A_num, G_ind, G_time, junk
    return Tc, RA_h, A_h

def generate_random_input(NNc):
    dw = random()*20 + 20
    dt = random()*0.07 + 0.03
    P = 5*dt + random()*0.12*dw
    G = dw*0.75*(1 + random())
    cr = (0.01 + random()*0.09)/dt #count rate
    if cr*G > 150:
        cr = cr*0.8
    dm = cp.random.random(4)*4-2
    dm = (cp.tanh(dm)+1)/2
    dm[0] += int(random()*1.5)*(2 + 3*random())
    dm = dm/cp.sum(dm)
    dm[dm<0.03]=0
    if(cp.sum(dm[2:4])<0.03):
        dm[2] = 0.03
    dm = dm/cp.sum(dm)
    
    mu0 = cp.arange(4)+1
    num0 = cp.dot(dm,mu0)
    fr = cr/num0 # fission rate
    T = NNc/cr #cycle time
        
    return dw,dt,fr,dm,T,P,G

NNc = 5*10**7

npdata = np.loadtxt('/kaggle/input/psmcmodel/PSMC_inp_2.csv',delimiter=',')

npdm = npdata[:,0:10]
npfr = npdata[:,10]
dm = cp.array(npdm)
fr = cp.array(npfr)

mu = cp.arange(10)+1
cr = cp.dot(dm[:,],mu)
cr[:] = cr[:]*fr[:]
T = NNc/cr[:]

Num_samples = fr.shape[0]
Num_cycles = (2.0e+9/T[:]+1).astype('int')

out0_all = cp.zeros((Num_samples,528))
out1_all = cp.zeros((Num_samples,528))
now = datetime.now()

dw = 48.0
dt = 0.095
P = 3.0
G = 60.0

for k in range(Num_samples):
    out0 = cp.zeros(513)
    out1 = cp.zeros(513)
    for i in range(int(Num_cycles[k])):
        pt0 = generate_pt0(dw, fr[k], dm[k,:], T[k])
        pt1 = apply_deadtime(pt0, 0.0)
        out0 += cp.hstack(calculate_raa(pt1, P, G, dw))
        pt1 = apply_deadtime(pt0, dt)
        out1 += cp.hstack(calculate_raa(pt1, P, G, dw))
    
    out0_all[k,:] = cp.hstack((dm[k,:],fr[k],dw,dt,P,G,out0))
    out1_all[k,:] = cp.hstack((dm[k,:],fr[k],dw,dt,P,G,out1))
    save1 = cp.savetxt('results_PSMC_2_dt_000.csv', out0_all.transpose())
    save1 = cp.savetxt('results_PSMC_2_dt_095.csv', out1_all.transpose())
    
    print(k, datetime.now()-now)
    now = datetime.now()
print("")