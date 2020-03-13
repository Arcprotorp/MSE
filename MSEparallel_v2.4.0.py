import numpy as np
import itertools
import time
from timeit import default_timer as timer
import scipy as scp
import scipy.stats as ssm
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
#from matplotlib import pyplot as plt
#import matplotlib.animation as animation


def NBtorus(X, r = 1, s = 1):
    #Might need to be reduced only to focus metabolic enzymes
    #Will return the neighbouring
    #print(X)
    a=np.arange(-1*r, r+1)
    district=itertools.product(a, a)
    #np.roll(np.roll(X, y, axis=0), x, axis=1)

    N=((1-s)*X)+(s*sum(np.roll(np.roll(X, pac[1], axis=0), pac[0], axis=1) for pac in district))
    #print(N.shape)
    return N


def random_rotate_windows(a,H,W):
    m,n,r = a.shape
    a5D = a.reshape(m//H,H,n//W,W,-1)
    cw0 = a5D[:,::-1,:,:,:].transpose(0,2,3,1,4)
    ccw0 = a5D[:,:,:,::-1,:].transpose(0,2,3,1,4)
    mask = np.random.choice([False,True],size=(m//H,n//W))
    out = np.where(mask[:,:,None,None,None],cw0,ccw0)
    return out.swapaxes(1,2).reshape(a.shape)



def vectorized_diffusion(a,H,W,pD):
    #pD - chance that a sub-array is rotated in a random direction
    rand_shift = np.random.randint(-1,2)
    rand_axis = np.random.randint(0,2)
    a = np.roll(a, shift = rand_shift, axis = rand_axis)
    # Since the 2*2 subgrid system is fixed, I decided to ocassionally
    #disturb the grid by rolling the whole array by one in a given
    #direction, as in my work the array is a toroid grid i considered every direction
    m,n,r = a.shape
    a5D = a.reshape(m//H,H,n//W,W,-1)
    cw0 = a5D[:,::-1,:,:,:].transpose(0,2,3,1,4)
    ccw0 = a5D[:,:,:,::-1,:].transpose(0,2,3,1,4)
    original = a5D[:,:,:,:,:].transpose(0,2,1,3,4)
    mask_clockdirection = np.random.choice([False,True],size=(m//H,n//W))
    mask_stationary = np.random.choice([True,False],size=(m//H,n//W), p=[1-pD,pD])
    w0 = np.where(mask_clockdirection[:,:,None,None,None],cw0,ccw0)
    out = np.where(mask_stationary[:,:,None,None,None],original,w0)
    out = out.swapaxes(1,2).reshape(a.shape)
    out_rerolled = np.roll(out, shift = -1*rand_shift, axis = rand_axis)
    #this way the disturbed grid is rerolled into its original position
    return out_rerolled



def performance(X,Y,R,M):
    #IF NBarray focuses only on metabolic enzymes
    #N Original array of cells
    #replicative=X[:,:,0:R]
    #metabolic=Y[:,:,R:M]
    #all useful=X[:,:,0:R+M]
    #V=np.dstack((replicative,metabolic))
    #V=np.concatenate((repl,meta),axis=2)
    #W=ssm.gmean(V,axis=2)
    #print("orig",X)
    #print("X\n",X[:,:,0:R]) #"""replicative set"""
    #print("Y\n",Y[:,:,R:R+M]) #"""metabolic set"""
    #print("X+Y\n",(np.dstack((X[:,:,0:R],Y[:,:,R:R+M])))) #"""effective set"""
    Wabs=ssm.gmean((np.dstack((X[:,:,0:R],Y[:,:,R:R+M]))),axis=2)
    Wabs[np.isnan(Wabs)]=0

    if np.max(Wabs)>0:
        #print(np.max(Wabs))
        #print("Wabs\n",Wabs)
        #print("Wrel\n",Wabs/np.max(Wabs))
        return (Wabs/np.max(Wabs))
    else:
        #print(np.max(Wabs))
        return np.zeros(Wabs.shape)



def WMAX(X,Y,R,M):
    #IF NBarray focuses only on metabolic enzymes
    #N Original array of cells
    #replicative=X[:,:,0:R]
    #metabolic=Y[:,:,R:M+1]
    #V=np.dstack((replicative,metabolic))
    #V=np.concatenate((repl,meta),axis=2)
    #W=ssm.gmean(V,axis=2)

    Wabs=ssm.gmean((np.dstack((X[:,:,0:R],Y[:,:,R:R+M]))),axis=2) #r+m+1???
    Wabs[np.isnan(Wabs)]=0

    return(np.max(Wabs))


def WABS(X,Y,R,M):
    #IF NBarray focuses only on metabolic enzymes
    #N Original array of cells
    #replicative=X[:,:,0:R]
    #metabolic=Y[:,:,R:M+1]
    #V=np.dstack((replicative,metabolic))
    #V=np.concatenate((repl,meta),axis=2)
    #W=ssm.gmean(V,axis=2)

    Wabs=ssm.gmean((np.dstack((X[:,:,0:R],Y[:,:,R:R+M]))),axis=2) #r+m+1???
    Wabs[np.isnan(Wabs)]=0

    return(Wabs)



def vectorized(prob_matrix, items):
    #print("MP\n",prob_matrix)
    s = prob_matrix.cumsum(axis=0)
    #print("s\n",s)
    r = np.random.rand(prob_matrix.shape[1])
    #print("r\n",r)
    #print("s<r\n",s<r)
    k = (s < r).sum(axis=0)
    #print("k\n",k)
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
    #print("NANrepweights\n",repweights)
    #repweights[np.isnan(repweights)]=0
    #print("repweights\n",repweights)
    temp = repweights.reshape(sqrsize*sqrsize,bins)
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



def decay_BIN(X,decay=0.001):
    #print("Work ahead")


    decadents = np.random.binomial(n=X.astype(int),p=decay)

    Y = X - decadents

    return Y



def division_OH(X,T,agro=0.01): #OVERAULED AND INTENDED DIVISION METHOD!
    #Work in progress
    dividers = (np.sum(X, axis=2)>T) #cells ower a threshold are primed for division
    #dormants = (dividers==False) #dormant areas
    alive = (np.sum(X, axis=2)>0) #alive areas

    cells_to_divide=(X * dividers[...,None]).astype(int) #cells which are dividing

    #print(np.sum(np.sum(np.sum(array_to_divide))))

    heirs = np.random.binomial(n=cells_to_divide,p=0.5) #3D after division, these cells will inherit the position
    knights = cells_to_divide - heirs #3D array of the remaining heritage, it is for the wandering knights
    #knights = remains * dividers[...,None] #knights acquire their heritage

    rand_shift = np.random.randint(-1,2)
    rand_axis = np.random.randint(0,2)

    #THIS LINE
    knight_arrives = np.roll(random_rotate_windows(np.roll(knights,shift=rand_shift,axis=rand_axis),2,2),shift=-1*rand_shift,axis=rand_axis)
    #####
    #^3D: A very puritan way for Brownian motion - every 2*2 sub-array turns by 90 degrees
    #the 2*2 grid is fixed, therefore in every round we roll the array,to mix up things

    knight_grid = (np.sum(knight_arrives, axis=2)>0) #2D grid where the knight arrives
    conflict_grid = alive * knight_grid #these are territories where the knights find inhabitated place, hence the conflict
    peace_grid = (conflict_grid==False) #places where no conflict happends

    knight_wins = conflict_grid * (np.random.random(size=dividers.shape)<agro) #2D array with values 1 where the knight wins

    knight_loses = (knight_wins==False)

    usurpers = knight_arrives * knight_wins[...,None]


    #usurped = X * knight_wins[...,None]
    peaceful_knights = knight_arrives * peace_grid[...,None]

    X = ((X - cells_to_divide + heirs) * knight_loses[...,None]) + usurpers + peaceful_knights

    return X


def mortalizer(X,W,pdie=0.1):
    dormants = (W==0)
    alive = (dormants==False)
    survive = np.random.binomial(n=dormants,p=1-pdie)

    nextgen = X * alive[...,None] + X * survive[...,None]

    return nextgen


def bristerna(X,R,M):
    metabolic_counts = np.count_nonzero(X[:,:,R:R+M],axis=2)
    has_allMi =  np.sum(metabolic_counts == M)
    has_someMi = X.shape[0]**2 - has_allMi - np.sum(metabolic_counts == 0)

    return(has_allMi, has_someMi)



#MAIN BODY
def MSE(param_set):
    start=timer()
    print("Initialize...")
    #M.S.E. - Metabolically connected Stochastic corrector Environment
    gridsize=100
    bin_num=20

    t=param_set[0] #treshold multiplier for division

    R=param_set[1] #replication enzyme set size
    
    M=param_set[2] #metabolic enzyme set size

    S=param_set[3]
    
    mu=param_set[4] #mutation probability
    
    rad_M=param_set[5] #metabolic neighbourhood
    
    STEPNUM=param_set[6]
    
    SEED=param_set[7]

    diffchance=param_set[8]


    #static vars
    agression=1      #diffchance/10 #chance that the occupied area will be usurped by a wandering knight
    enzymedecay=0      #0.001
    death=1     #0.1
    e=0.00000000000001
    T=(R+M)*t
    startgenenum=round(t/2)

    #Calculating the ultifitness
    L=(((2*rad_M)+1)**2)-1
    nimax=(((S*L)+1)*T)/(R+M)

    if (R*nimax)>T:
        rimax=T/R
        mimax=(S*L*T)/M
        deus_Wult=((rimax**R)*(mimax**M))**(1/(R+M))
    else:
        deus_Wult=nimax


    #supercell=np.array([(T/(R+M))/T for i in range(0,R+M)])
    #maxW=ssm.gmean(supercell)
    MSG = ("MSE"+"_gridsize="+str(gridsize)+
           "_t="+str(t)+"_mu="+str(mu)+
           "_R="+str(R)+"_M="+str(M)+
           "_radM="+str(rad_M)+
           "_S="+str(S)+
           "_diff="+str(diffchance)+"_seed="+str(SEED))
    print(MSG)


    FILE=("MSEv2.4.0"+"_gridsize="+str(gridsize)+
          "_t="+str(t)+"_mu="+str(mu)+
          "_R="+str(R)+"_M="+str(M)+
          "_radM="+str(rad_M)+
          "_S="+str(S)+"_Wult="+str(round(deus_Wult,3))+
          "_diff="+str(diffchance)+"_Edecay="+str(enzymedecay)+
          "_steps="+str(STEPNUM)+"_seed="+str(SEED))

    file=open(FILE+".txt", "w")

    #histograms
    histnum=bin_num
    histclass=bin_num+1
    static_histnum=bin_num
    static_histclass=bin_num+1
    
    header=("step\tpop\tWmax\tnR\tnM\tnD\tnAU\tnSP\tFP\t"+
            "\t".join("F"+str(i) for i in range(1,bin_num+1))+"\t"+
            "\t".join("bin"+str(i) for i in range(1,bin_num+2))+"\t"+
            "\t".join("Wult_F"+str(i) for i in range(1,bin_num+1))+"\t"+
            "\t".join("Wult_bin"+str(i) for i in range(1,bin_num+2))+"\t"+
            "\t".join(list("R"+str(i) for i in range(1,R+1))+
                      list("M"+str(i) for i in range(1,M+1)))+"\n")

    file.write(header)
    file.close()

    
    #SETTING UP MSE-environment
    np.random.seed(SEED)
    #start = timer()
    #MSE=np.random.randint(startmax, size=(gridsize, gridsize,R+M+1)).astype(int)#*np.random.binomial(n=1,p=0.25,size=(gridsize,gridsize))[...,None].astype(int)
    MSE=np.ones(shape=(gridsize, gridsize,R+M+1))*startgenenum
    MSE[:,:,-1]=np.zeros(shape=(gridsize, gridsize)) #Layer of mutants starts with 0

    N = NBtorus(MSE, r = rad_M, s=S)
    x = performance(MSE, N, R, M)

    if True:
        for step in range(STEPNUM+1):
            #Diffusion step
            MSE = vectorized_diffusion(MSE,2,2,pD=diffchance)
            
            #Neighbourhood, and fitness calc
            N = NBtorus(MSE, r = rad_M, s=S)
            Wrel = performance(MSE, N, R, M)
            Wmax = WMAX(MSE, N, R, M)

            #print(step,":",Wmax)

            #DEATH COMETH
            MSE = mortalizer(X=MSE,W=Wrel,pdie=death)

            #BUT "Life, Uh, Finds a Way"
            MSEplus = replication(MSE ,Wrel, sqrsize = gridsize, mu=mu, bins=R+M+1)

            #standard decay procedure
            MSEminus = decay_BIN(MSEplus,decay=enzymedecay)

            #rep = replication(Wrel, sqrsize = gridsize, enz_num = R+M+1)
            #dec = decay(Wrel, sqrsize = gridsize, enz_num = R+M+1)
            MSE = division_OH(MSEminus, T=T, agro=agression)
            
            if step%100==0:
                file=open(FILE+".txt", "a")
                cellsizes=np.sum(MSE, axis=2)
                pop=np.sum(cellsizes>0,axis=(0,1))
                #2DloadMap=(np.sum(MSE[:,:,0:R+M], axis=2)/(np.sum(MSE, axis=2)+e))
                ultload=(np.sum(MSE[:,:,0:R+M])/(np.sum(MSE)+e))

                wabbit_season=WABS(MSE, N, R, M)
                
                Whist=np.histogram(wabbit_season,bins=bin_num)
                
                Wult_hist=np.histogram((wabbit_season/deus_Wult),range=(0,1),bins=bin_num)
                
                nR=int(np.sum(MSE[:,:,0:R]))
                nM=int(np.sum(MSE[:,:,R:R+M]))
                nD=int(np.sum(MSE[:,:,-1]))

                #v.2.3.1
                lifestyles=bristerna(MSE,R,M)
                nAU=lifestyles[0]
                nSP=lifestyles[1]
                nFP=pop-nAU-nSP
                Ris=(int(np.sum(MSE[:,:,i])) for i in range(0,R))
                Mis=(int(np.sum(MSE[:,:,R+i])) for i in range(0,M))

                data=([step,pop,(round(Wmax,3)),nR,nM,nD,nAU,nSP,nFP]+
                      list(Whist[0])+list(round(w,3) for w in Whist[1])+
                      list(Wult_hist[0])+list(round(wult,3) for wult in Wult_hist[1])+
                      list(Ris)+list(Mis))
                file.write("\t".join(str(x) for x in data)+"\n")
                file.close()
                
                if pop<50:
                    print("Sorry kiddo, you seem kinda weak")
                    file=open(FILE+".txt", "a")
                    fin=timer()-start
                    file.write("END\t"+str(round(fin,0))+"\n")
                    file.close()
                    print ("Ended an MSE run")
                    return

    #Putin' an END to this
    file=open(FILE+".txt", "a")
    fin=timer()-start
    file.write("END\t"+str(round(fin,0))+"\n")
    file.close()
    print ("Ended an MSE run")
    return


#Start
if __name__ == '__main__':

    t = [50,100] #treshold for division
    R = [2,5,10] #replication enzyme set size
    M = [2,5,10] #metabolic enzyme set size
    S = [0.5,1]
    mu = [0.1,0.01,0.001] #mutant probability mu~[0.01,0.001,0.0001] l=200
    rad_M = [1,2,3] #metabolic neighbourhood
    STEPNUM = [1000000]
    SEED = [17,23,42]
    diffchance = [0.1,0.01,0.001]
    
    paramlen=np.prod(list(len(i) for i in (t,R,M,S,mu,rad_M,STEPNUM,SEED,diffchance)))

    param_space = itertools.product(t,R,M,S,mu,rad_M,STEPNUM,SEED,diffchance)

    #NO NEIGHBOURS
    t2 = [50,100] #treshold for division
    R2 = [3,6,9,11,14,19] #replication enzyme set size
    M2 = [1] #metabolic enzyme set size
    S2 = [0]
    mu2 = [0.1,0.01,0.001] #mutant probability mu~[0.01,0.001,0.0001] l=200
    rad_M2 = [1] #metabolic neighbourhood
    STEPNUM2 = [1000000]
    SEED2 = [17,23,42]
    diffchance2 = [0.1,0.01,0.001]
    
    paramlen2=np.prod(list(len(j) for j in (t2,R2,M2,S2,mu2,rad_M2,STEPNUM2,SEED2,diffchance2)))

    print("Pooling MSE for",(paramlen+paramlen2),"runs with",mp.cpu_count()-2,"cores")

    param_space2 = itertools.product(t2,R2,M2,S2,mu2,rad_M2,STEPNUM2,SEED2,diffchance2)

    PARAM_PARAM=itertools.chain(param_space,param_space2)


    #p = mp.Pool(1) #Only one cpu
    #p = mp.Pool(mp.cpu_count()-1) #MAX-1 cpu
    p = mp.Pool(mp.cpu_count()-2) #number of CPU core to use (here: all of them)


    p.map(MSE, PARAM_PARAM)


    print("SIMULATION IS OVER!")

    #time.sleep(600)
####USAGE
####
####works only in command line
####cd to library where MSEparallel is located
####then:
####python MSEparallel_vX.Y.py


#Some interesting settings
#MSEv2.3.2_gridsize=100_t=50_mu=0.0198_R=2_M=5_radM=1_S=0.5_Wult=250.0_diff=0.1_Edecay=0_steps=1000000_seed=17
