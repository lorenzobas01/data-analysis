import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import random as rdm

#This code generates random numbers from 0 to an imput number and performs a data analysis of this number distribution

num=int(input('How many numbers is the distribution formed by: ')) 
fin_number=int(input('Final numner of distribution: '))

data=np.zeros(num)
#Now we fill our set of data
for i in range(num):
    data[i]=rdm.uniform(0,fin_number)

mean=np.mean(data)    #mean value


fun=data-mean*np.ones(len(data))

var=np.mean(np.square(fun)) #variance

std=np.sqrt(var) #standard deviation

skw=np.mean(np.power(fun,3))/np.power(np.mean(np.power(fun,2)),1.5) #skewness

krt=(np.mean(np.power(fun,4))/(np.power(np.mean(np.power(fun,2)),2)))-3 #Kurtosis

pdf_data=sps.gaussian_kde(data) 

data_range=np.linspace(np.min(data),np.max(data),num)
pdf=pdf_data(data_range) #This is our probability density funcion


cdf=0*np.ones(len(pdf)) 

for i in range(1,num):
    cdf[i]=np.trapz(pdf[0:i+1],data_range[0:i+1])




normal_distribution=sps.norm.pdf(data_range,mean,std) #we can compare our pdf with a gaussian distribution with the seam mean value and standard deviation


plt.figure(1)
plt.plot(data_range,normal_distribution,data_range,pdf)
plt.text(0.8, 0.8, f'Mean: {mean:.2f}\nStd Dev: {std:.2f}\nKurtosis: {krt:.2f}', transform=plt.gca().transAxes)
plt.ylabel('pdf vs normal distribution')
plt.figure(2)
plt.plot(data_range,cdf)
plt.ylabel('cdf')
plt.show()

