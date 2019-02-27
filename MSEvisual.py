import numpy as np
import itertools
from timeit import default_timer as timer
import scipy as scp
import scipy.stats.mstats as ssm
from matplotlib import pyplot as plt
import matplotlib.animation as animation



def NBtorus(X, r = 1):
    #Might need to be reduced only to focus metabolic enzymes
    #Will return the neighbouring
    a=np.arange(-1*r, r+1)
    district=itertools.product(a, a)
    #np.roll(np.roll(X, y, axis=0), x, axis=1)

    N=np.sum(np.roll(np.roll(X, pac[1], axis=0), pac[0], axis=1) for pac in district)

    return N

def performance(X,Y,R,M):
    #IF NBarray focuses only on metabolic enzymes
    #N Original array of cells
    #replicative=X[:,:,0:R]
    #metabolic=Y[:,:,R:M+1]
    #V=np.dstack((replicative,metabolic))
    #V=np.concatenate((repl,meta),axis=2)
    #W=ssm.gmean(V,axis=2)

    Wabs=ssm.gmean((np.dstack((X[:,:,0:R],Y[:,:,R:M+1]))),axis=2)

    return (Wabs/np.max(Wabs))

def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    #print(k)
    return items[k]

def binom(x):
    fis=int(np.random.binomial(x,p=0.5,size=1))
    return fis

def replication(X,W,sqrsize,mu,bins):
    
    #print("Work ahead")
    randcomp = np.random.random_sample(size=(sqrsize,sqrsize))
    reparray = (W>randcomp) #This will tell us which cells will replicate one enzyme
    mutarray = np.random.binomial(n=1, p=mu, size=(sqrsize,sqrsize)) #Cells which might be mutants

    repweights = X/np.sum(X,axis=2)[..., None] #probability of chosing the enzymegroup to copy

    temp = repweights.reshape(gridsize*gridsize,bins)
    temp = np.swapaxes(temp,0,1)
    
    replicator_indices=vectorized(temp, np.arange(bins)).reshape(sqrsize,sqrsize)

    #print(replicator_indices)

    chosenones = np.zeros(shape=(sqrsize,sqrsize,bins)) #defines which genes were chosen to replicate

    #print(chosenones)
    
    a, b = np.ogrid[:sqrsize, :sqrsize]
    chosenones[a, b, replicator_indices] = 1 #Will define exactly which gene was chosen and in which cell

    
    addmutant = (reparray==True) & (mutarray==1) #2D array of cells who copied producing nonfunctional mutant

    fidelcells = (reparray==True) & (mutarray==0) #2D array of cells who copied perfectly

    addnormal = chosenones * fidelcells[..., None] #3D array of genes(in cells) which were actially copied
    
    Y=np.copy(X)
    Y[:,:,-1] = Y[:,:,-1] + addmutant

    Y = Y + addnormal
    #print (Y)
    return Y #Cellular environment with replicated enzymes

    #reparray=1*np.random.randint(2,size=(sqrsize,sqrsize,enz_num))


def decay(X,W,sqrsize,bins):
    #print("Work ahead")
    randcomp=np.random.random_sample(size=(sqrsize,sqrsize))
    decarray= (W<randcomp) | np.isnan(W) #This will tell us which cells will lose one enzyme due to decay

    hasany=(np.sum(X,axis=2)>0) #cells actually containing something
    
    decweights = X/np.sum(X,axis=2)[..., None] #probability of chosing the enzymegroup to decay

    temp = decweights.reshape(gridsize*gridsize,R+M+1)
    temp = np.swapaxes(temp,0,1)

    decay_indices=vectorized(temp, np.arange(bins)).reshape(gridsize,gridsize)

    chosenones = np.zeros(shape=(sqrsize,sqrsize,bins)) #defines which genes were chosen to replicate
    
    a, b = np.ogrid[:sqrsize, :sqrsize]
    chosenones[a, b, decay_indices] = 1 #Will define exactly which gene was chosen and in which cell

    decaycells = (decarray==True) & (hasany==True)

    decadents = chosenones * decaycells[..., None]

    Y=np.copy(X)

    Y = Y - decadents

    return Y

def division(X,T):
    #Work in progress
    dividers = (np.sum(X, axis=2)>T)
    dormants = (dividers==False)

    cells_to_divide=(X * dividers[...,None]).astype(int) #cells which are dividing

    #print(np.sum(np.sum(np.sum(array_to_divide))))

    heirs = np.random.binomial(n=cells_to_divide,p=0.5) #3D after division, these cells will inherit the position
    remains = cells_to_divide - heirs #3D array of the remaining heritage, it is for the wandering knights

    
    knights = remains * dividers[...,None] #knights acquire their heritage
    
    knight_lands = np.roll(knights,shift=np.random.choice([1,-1], p=[0.5, 0.5]),axis=np.random.randint(0,2))
    #^3D: This is just a deterministic shift in one direction! All knights will move together in one step
    
    knight_grid = (np.sum(knight_lands, axis=2)>0)#2D array where the knights wandered

    residuals = (X * dormants[...,None]) * (knight_grid==False)[...,None]
    #^3D array: those who will not divide, but previous owners of where a knight arrives will be murdered
    
    Y = residuals + heirs + knight_lands
    
    return Y


#M.S.E. - Metabolically connected Stochastic corrector Environment
gridsize=300
T=50 #treshold for division
R=2 #replication enzyme set size
M=3 #metabolic enzyem set size
mu=0.0001 #mutation probability
rad_M=5 #metabolic neighbourhood
STEPNUM=1000

#supercell=np.array([(T/(R+M))/T for i in range(0,R+M)])
#maxW=ssm.gmean(supercell)

FILE=("MSE"+"_gridsize="+str(gridsize)+
      "_T="+str(T)+"_mu="+str(mu)+
      "_R="+str(R)+"_M="+str(M)+
      "_radM="+str(rad_M)+"_steps="+str(STEPNUM))

np.random.seed(42)
start = timer()
MSE=np.random.randint(5, size=(gridsize, gridsize,R+M+1))
MSE[:,:,-1]=np.zeros(shape=(gridsize, gridsize)) #Layer of mutants starts with 0

FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title=FILE, artist='Arcprotorp')
writer = FFMpegWriter(fps=30, metadata=metadata)
fig = plt.figure()
fig.patch.set_facecolor('black')
N = NBtorus(MSE, r = rad_M)
x = performance(MSE, N, R, M)

with writer.saving(fig, FILE+".mp4", 50):
    plt.spy(x)
    plt.axis('off')
    writer.grab_frame()
    plt.clf()    
    for step in range(STEPNUM):
        print(step)
        N = NBtorus(MSE, r = rad_M)
        Wrel = performance(MSE, N, R, M)

        MSEplus = replication(MSE ,Wrel, sqrsize = gridsize, mu=mu, bins=R+M+1)
        MSEminus = decay(MSEplus,Wrel,sqrsize = gridsize, bins=R+M+1)
        #rep = replication(Wrel, sqrsize = gridsize, enz_num = R+M+1)
        #dec = decay(Wrel, sqrsize = gridsize, enz_num = R+M+1)
        MSE = division(MSEminus,T=T)
        #Visualizer submodule
        x = performance(MSE, N, R, M)
        plt.spy(x)
        plt.axis('off')
        plt.text(1,1,"step="+str(step), fontsize=20)
        writer.grab_frame()
        plt.clf()

#print (MSE)

end = timer()
print(end - start)