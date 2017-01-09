import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from scipy import stats
import random

"""
In the following code we build a null signal class, that makes linear combinations of noise coming from 3 pulsars.
"""
class NullSignal:
    def __init__(self, i, j, k, Noise):
        self.SinPhi = np.sin(2 * phi)
        self.CosPhi = np.cos(2 * phi)
        self.Noise = Noise
        self.Degree = (180/np.pi)
        self.PulsarLat = np.array([PulsarDataArray[i,0],PulsarDataArray[j,0],PulsarDataArray[k,0]])/self.Degree
        self.PulsarLong = np.array([PulsarDataArray[i,1],PulsarDataArray[j,1], PulsarDataArray[k,1]])/self.Degree
        self.PulsarError = np.array([PulsarDataArray[i,3],PulsarDataArray[j,3], PulsarDataArray[k,3]])*1e-9
        self.SinPulsLat = np.sin(self.PulsarLat)
        self.SinPuls2Lat = np.sin(2 * self.PulsarLat)
        self.SinPulsLong = np.sin(self.PulsarLong)
        self.CosPulsLat = np.cos(self.PulsarLat)
        self.CosPuls2Lat = np.cos(2*self.PulsarLat)
        self.CosPulsLong = np.cos(self.PulsarLong)

    def NoiseResidual(self,t,i):
        self.L = len(t)

        return np.random.normal(0.0 , self.PulsarError[i], self.L)


    def B1(self, SkyLat, SkyLong, i):
        return ((1 + np.sin(SkyLat[:,np.newaxis]) ** 2) * self.CosPulsLat[i] ** 2 * np.cos(2 * ( SkyLong - self.PulsarLong[i]))
            - np.sin(2. * SkyLat[:,np.newaxis]) * self.SinPuls2Lat[i] * np.cos(SkyLong - self.PulsarLong[i])
            + (2. - 3. * self.CosPulsLat[i] ** 2) * np.cos(SkyLat[:,np.newaxis]) ** 2)

    def B2(self, SkyLat, SkyLong, i):
        return (2. * np.cos(SkyLat[:,np.newaxis]) * np.sin(2 * self.PulsarLat[i]) * np.sin(SkyLong - self.PulsarLong[i])
            - 2 * np.sin(SkyLat[:,np.newaxis]) * np.cos(self.PulsarLat[i]) ** 2 * np.sin(2 * ( SkyLong - self.PulsarLong[i])))


    def aCoefficients(self, SkyLat, SkyLong):
        FMatrix=np.array([
                    [self.B1(SkyLat[:,np.newaxis], SkyLong, 0)*self.CosPhi+self.B2(SkyLat[:,np.newaxis], SkyLong, 0)*self.SinPhi,
                    self.B1(SkyLat[:,np.newaxis], SkyLong, 1)*self.CosPhi+self.B2(SkyLat[:,np.newaxis], SkyLong, 1)*self.SinPhi,
                    self.B1(SkyLat[:,np.newaxis], SkyLong, 2)*self.CosPhi+self.B2(SkyLat[:,np.newaxis], SkyLong, 2)*self.SinPhi]
                    ,
                    [self.B2(SkyLat[:,np.newaxis], SkyLong, 0)*self.CosPhi-self.B1(SkyLat[:,np.newaxis], SkyLong, 0)*self.SinPhi,
                    self.B2(SkyLat[:,np.newaxis], SkyLong, 1)*self.CosPhi-self.B1(SkyLat[:,np.newaxis], SkyLong, 1)*self.SinPhi,
                    self.B2(SkyLat[:,np.newaxis], SkyLong, 2)*self.CosPhi-self.B1(SkyLat[:,np.newaxis], SkyLong, 2)*self.SinPhi]
                    ])
        return np.array([
                    FMatrix[0,1,:,0,:]*FMatrix[1,2,:,0,:]-FMatrix[0,2,:,0,:]*FMatrix[1,1,:,0,:],
                    FMatrix[0,2,:,0,:]*FMatrix[1,0,:,0,:]-FMatrix[0,0,:,0,:]*FMatrix[1,2,:,0,:],
                    FMatrix[0,0,:,0,:]*FMatrix[1,1,:,0,:]-FMatrix[0,1,:,0,:]*FMatrix[1,0,:,0,:],
                    ])

    def NullSignalNoise(self, t, SkyLat, SkyLong):
        self.aNull=self.aCoefficients(SkyLat[:,np.newaxis], SkyLong)
        Sig = (self.aNull[0,:,:]*self.NoiseResidual(t, 0)[np.newaxis,:,np.newaxis] + self.aNull[1,:,:]*self.NoiseResidual(t, 1)[np.newaxis,:,np.newaxis] + self.aNull[2,:,:]*self.NoiseResidual(t, 2)[np.newaxis,:,np.newaxis])
        return Sig

PulsarPath='/Users/jeffrey/GoogleDrive/NullSignalNew/NANOGravPulsarSkyLocations_Lat_Long_Name_Noise.csv'
PulsarDataArray= np.genfromtxt(PulsarPath, delimiter=",")

""" Iitialize all of the counting arrays for sky maps and iterations. """
LatRange = 90
LongRange = 360
Latitude = np.linspace(-LatRange/180.*np.pi, LatRange/180.*np.pi, 2*LatRange, endpoint=False)
Longitude = np.linspace(0, LongRange/180.*np.pi, LongRange, endpoint=False)

iterations = np.arange(100)
IterLength = len(iterations)

phi = 0 #Polarization angle of the pulsars
NoiseCoeff = 1 #Here we pull noises from the csv file, but if not we need this noise coefficient.

T0=time.time()

PulsarIndex = np.arange(0, 18)
PulsarCombo =  list(itertools.combinations(PulsarIndex, 3))
"""Makes a list of all combinations of 18 choose 3.
"""

filename = "NoiseRealization_P18_NG_Noise.dat"
Total = np.memmap(filename, dtype = 'float32', mode = 'w+', shape = (180,360,10000))
#Initialize files

T4 = time.time() #Time stamp for keeping track of the iterations.
print("Time Check #0:", T4-T0) #See how long the file intialization takes.

for Round in range(0, 10000//IterLength):
    #Loop of 100 sky maps at a time in an array.
    ProductIter = np.zeros((len(Latitude),IterLength,len(Longitude)))

    for Iter in range(0 , len(PulsarCombo)):
        NS1 = NullSignal(PulsarCombo[Iter][0],PulsarCombo[Iter][1],PulsarCombo[Iter][2], NoiseCoeff)
        NoiseIter = np.log((NS1.NullSignalNoise(iterations, Latitude, Longitude))**2)
        ProductIter += NoiseIter
        #Make a sum of all of the null signals that are made up of the different combinations of pulsars.
        del NS1
        del NoiseIter

    Product = np.transpose(ProductIter,(0,2,1)) #Take the transpose (artifact of other code)
    del ProductIter
    T2 = time.time()
    print("Round:",Round, " Time Check #1:", T2-T0)

    ii = Round*IterLength
    jj = Round*IterLength + IterLength

    Total[:,:,ii:jj] = Product #Add these 100 iterations to the file in storage. 
    
    del Product
    T3 = time.time()
    print("Round:",Round, " Time Check #2:", T3-T0)
del Total

T1 = time.time()
print(T1-T0)
