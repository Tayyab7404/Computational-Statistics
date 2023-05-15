#!/usr/bin/env python
# coding: utf-8

# ## Functions, Import Statements:

# In[33]:


import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as ss
import pandas as pd

def mylen(data):
    size = 0
    for i in data:
        size = size + 1
    return size

def mysum(data):
    s = 0
    for i in data:
        s = s + i
    return s

def mymean(data):
    Sum = 0
    for i in data:
        Sum = Sum + i
    return Sum/mylen(data)

def DET(matrix):
    Sum = 0
    
    Sum = Sum + matrix[0][0] * (matrix[1][1]*matrix[2][2] - matrix[2][1]*matrix[1][2])
    Sum = Sum - matrix[0][1] * (matrix[1][0]*matrix[2][2] - matrix[2][0]*matrix[1][2])
    Sum = Sum + matrix[0][2] * (matrix[1][0]*matrix[2][1] - matrix[2][0]*matrix[1][1])
    
    return Sum
    
def CovXY(x, y):
    n = mylen(x)
    meanX = mymean(x)
    meanY = mymean(y)
    
    SumXY = mysum(np.multiply(x, y))
    
    CovXY = SumXY/n - meanX*meanY
    
    return CovXY
    
def SD(x):
    n = mylen(x)
    meanx = mymean(x)
    
    Sumx2 = mysum(np.multiply(x, x))
    
    Var = Sumx2/n - meanx**2
    
    return math.sqrt(Var)


# ## WEEK 1:
# #### Write a python program to find the best fit straight line of the form y = a+bx and draw the scatter plot.

# In[32]:


# Straight Line Best Fit:

def drawPlotLine(x, y, Ycal):
    plt.plot(x, y)
    plt.plot(x, Ycal)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Best Fit Line")
    plt.show()

def LinearRegression(x, y):
    n = mylen(x)
    m = mylen(y)
    
    if n != m:
        print("Invalid Data!")
        return
    
    slope = (n*mysum(np.multiply(x,y)) - mysum(x)*mysum(y)) / (n*mysum(np.multiply(x,x)) - mysum(x)**2)
    
    constant = (mysum(y) - slope*mysum(x)) / n
    
    Ycal = np.add(np.multiply(slope,x),constant)
    
    meanY = mymean(y)
    
    SST = mysum(np.array([(i-meanY)**2 for i in y]))
    SSE = mysum(np.multiply(np.subtract(y, Ycal),np.subtract(y, Ycal)))
    SSR = mysum(np.array([(i-meanY)**2 for i in Ycal]))
    
    Rsq = SSR/SST
    # Rsq = 1 - SSE/SST
    
    print("X values: ",x)
    print("Y values: ",y)
    print("Slope (m): ",slope)
    print("Constant (c): ",constant)
    print("Y Calculated values: ",Ycal)
    print("The Linear Regression Line for the given data:")
    print(f"Y = {slope}X + {constant}")
    print("Sum of Squares due to Total: ",SST)
    print("Sum of Squares due to Treatments: ",SSR)
    print("Sum of Squares due to Error: ",SSE)
    print("R Square value: ",Rsq)
    
    drawPlotLine(x, y, Ycal)
    
    if Rsq > 0.9:
        print("The Y Calculated values is the best fit for the given data")
    else:
        print("The Y Calculated values is not the best fit for the given data")
        
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

# x = np.array([1, 2, 3, 4, 6, 8])
# y = np.array([2.4, 3, 3.6, 4, 5, 6])

LinearRegression(x, y)


# ## WEEK 2:
# #### Write a python program to fit a second degree parabola of the form y = a+bx+cx2 and draw the scatter plot.

# In[31]:


# Parabola Best Fit:

def drawPlotParabola(x, y, Ycal):
    plt.plot(x, y)
    plt.plot(x, Ycal)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Best Fit Curve")
    plt.show()

def ParabolaRegression(x, y):
    n = mylen(x)
    m = mylen(y)

    if n != m:
        print("Invalid Data: ")
        return
    
    x2 = np.multiply(x, x)
    x3 = np.multiply(x2, x)
    x4 = np.multiply(x3, x)
    xy = np.multiply(x, y)
    x2y = np.multiply(x2, y)
    
    Sumx = mysum(x)
    Sumx2 = mysum(x2)
    Sumx3 = mysum(x3)
    Sumx4 = mysum(x4)
    Sumy = mysum(y)
    Sumxy = mysum(xy)
    Sumx2y = mysum(x2y)
    
    delta  = DET([[n, Sumx, Sumx2],
                  [Sumx, Sumx2, Sumx3],
                  [Sumx2, Sumx3, Sumx4]])
                  
    delta1 = DET([[Sumy, Sumx, Sumx2],
                  [Sumxy, Sumx2, Sumx3],
                  [Sumx2y, Sumx3, Sumx4]])
                  
    delta2 = DET([[n, Sumy, Sumx2],
                  [Sumx, Sumxy, Sumx3],
                  [Sumx2, Sumx2y, Sumx4]])
                  
    delta3 = DET([[n, Sumx, Sumy],
                  [Sumx, Sumx2, Sumxy],
                  [Sumx2, Sumx3, Sumx2y]])
    
    a = delta1 / delta
    b = delta2 / delta
    c = delta3 / delta
    
    Ycal = np.add(a, np.add(np.multiply(b, x), np.multiply(c, x2)))
    
    meanY = mymean(y)
    
    SST = mysum(np.array([(i-meanY)**2 for i in y]))
    SSE = mysum(np.multiply(np.subtract(y, Ycal),np.subtract(y, Ycal)))
    SSR = mysum(np.array([(i-meanY)**2 for i in Ycal]))
    
    Rsq = SSR/SST
    # Rsq = 1 - SSE/SST
    
    print("X values: ",x)
    print("Y values: ",y)
    print("a = ",a)
    print("b = ",b)
    print("c = ",c)
    print("Y Calculated values: ",Ycal)
    print("The Parabola Regression Line for the given data: ")
    print(f"Y = {a} + {b}X + {c}X2")
    print("Sum of Squares due to Total: ",SST)
    print("Sum of Squares due to Treatments: ",SSR)
    print("Sum of Squares due to Error: ",SSE)
    print("R Square value: ",Rsq)
    
    drawPlotParabola(x, y, Ycal)
    
    if Rsq > 0.9:
        print("The Y Calculated values is the best fit for the given data")
    else:
        print("The Y Calculated values is not the best fit for the given data")

x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

# x = np.array([2, 4, 6, 8, 10])
# y = np.array([3.07, 12.85, 31.47, 57.38, 91.29])

ParabolaRegression(x, y)


# ## WEEK 3:
# #### Write a python program to find Karl Pearson’s correlation coefficient between X and Y variables.

# In[29]:


# Karl Pearson's Correlation Coefficient:
    
def KPCC(x, y):
    if mylen(x) != mylen(y):
        print("Invalid Data!")
        return
    
    Covxy = CovXY(x,y)
    SDx = SD(x)
    SDy = SD(y)
    
    KPCC = Covxy / (SDx * SDy)
    
    print("Co-variance of X and Y = ",Covxy)
    print("Standard Deviation of X = ",SDx)
    print("Standard Deviation of Y = ",SDy)
    
    if KPCC > -1 and KPCC < 1:
        print(f"The Karl Pearson's Correlation Coefficient of the given data is: {KPCC}")
    else:
        print("Invalid Data!")
    
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

# x = np.array([16,21,26,23,28,24,17,22,21])
# y = np.array([33,38,50,39,52,47,35,43,41])

KPCC = KPCC(x, y)


# ## WEEK 4:
# #### Write a python program to find the Spearman’s correlation coefficient between X and Y variables.

# In[30]:


# Spearman's Rank Correlation Coefficient:

def Rankify(data):
    N = len(data)
    
    Ranks = [None for i in range(N)]
 
    for i in range(N):
        r = 1
        s = 0
 
        # Count no of bigger elements from 0 to i-1
        for j in range(i):
            if (data[j] > data[i]):
                r += 1
            if (data[j] == data[i]):
                s += 1
 
        # Count no of bigger elements from i+1 to N-1
        for j in range(i+1, N):
            if (data[j] > data[i]):
                r += 1
            if (data[j] == data[i]):
                s += 1
 
        # Use Fractional Rank formula
        # fractional_rank = r + (n-1)/2
        Ranks[i] = r + s*0.5

    return Ranks

def RemDups(data):
    Dup_data = []
    
    for i in data:
        if i not in Dup_data:
            Dup_data.append(i)
            
    return Dup_data

def CF(x,y):
    cf = 0
    
    x = list(x)
    y = list(y)
    
    Dup_x = RemDups(x)
    Dup_y = RemDups(y)
    
    for i in Dup_x:
        count = x.count(i)
        if count > 1:
            cf += (count * (count**2 - 1)) / 12
            
    for i in Dup_y:
        count = y.count(i)
        if count > 1:
            cf += (count * (count**2 - 1)) / 12
    
    return cf

def SRCC(x, y):
    n = len(x)
    m = len(y)

    if n != m:
        print("Invalid Data: ")
        return
    
    RankX = np.array(Rankify(x))
    RankY = np.array(Rankify(y))
    
    # Difference of Ranks:
    Di = np.subtract(RankX, RankY)
    DiSq = np.square(Di)
    
    SumDiSq = mysum(DiSq)
    
    # Correction Factor:
    cf = CF(x,y)
    
    SumDiSq += cf
    
    SRCC = 1 - ((6 * SumDiSq) / (n * (n**2 - 1)))
    
    data = {
        "X values": x,
        "Y values": y,
        "Ranks of X": RankX,
        "Ranks of Y": RankY,
        "Di values": Di,
        "Di sq. values": DiSq,
    }
    
    dataframe = pd.DataFrame(data)
    print(dataframe)
    
    print("Correction Factor: ",cf)
    print("Sum of Di sq. = ",SumDiSq)
    print(f"The Spearman's Ranked Correlation Coefficient of the given data = {SRCC}")
    
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

# x = np.array([68, 64, 75, 50, 64, 80, 75, 40, 55, 64])
# y = np.array([62, 58, 68, 45, 81, 60, 68, 48, 50, 70])

SRCC(x, y)


# ## WEEK 5:
# #### Write a python program to classify the data based on one way ANOVA.

# In[27]:


# ANOVA One Way Classification:

k = int(input("Enter the number of Treatments: "))
name = input("Enter name of the Treatments: ")

Treatments = []

# Treatments = [[90, 82, 79, 98, 83, 91], [105, 89, 93, 104, 89, 95, 86], [83, 89, 80, 94]]

for i in range(k):
    a = np.array([float(j) for j in input(f"Enter {name} {i+1} values: ").strip().split()])
    Treatments.append(a)
    
alpha = float(input("Enter Level of Significance: "))

Ti = Ti2 = RSS = N = 0

for i in Treatments:
    Ti += mysum(i)
    Ti2 += mysum(i)**2 / mylen(i)
    RSS += mysum(np.square(i))
    N += mylen(i)

CF = Ti**2 / N
SST = RSS - CF
SSTr = Ti2 - CF
SSE = SST - SSTr
MeanSSTr = SSTr / (k-1)
MeanSSE = SSE / (N-k)

Fcal = MeanSSTr / MeanSSE

if(Fcal < 1):
    Fcal = 1 / Fcal

FTable = ss.f.ppf(1-alpha, k-1, N-k)

print(f"Row Sum of Squares (RSS): {RSS}")
print(f"Correction Factor (CF): {CF}")
print("Grand Total (G): ",Ti)
print("Sum of Ti2/ni: ",Ti2)
print(f"Sum of Squares due to Total (SST): {SST}")
print(f"Sum of Squares due to Treatments (SSTr): {SSTr}")
print(f"Sum of Squares due to Error (SSE): {SSE}")
print(f"Mean Sum of Squares due to Treatments (Mean SST): {MeanSSTr}")
print(f"Mean Sum of Squares due to Error (Mean SSE): {MeanSSE}\n")

data = {
    "S O V": [name, "Error", "Total"],
    "S O S": ["{:.4f}".format(SSTr), "{:.4f}".format(SSE), "{:.4f}".format(SST)],
    "D O F": [k-1, N-k, N-1],
    "M S O S": ["{:.4f}".format(MeanSSTr), "{:.4f}".format(MeanSSE), " - "],
    "V R": ["F - cal = {:.4f}".format(Fcal),"~ F(K-1, N-K)",""]
}

df = pd.DataFrame(data)

print(df) 

print(f"\nF-Calculated Value: {Fcal}")
print(f"F-Table Value: {FTable}")

if(Fcal < FTable):
    print(f"We Accept H0\nThere is Homogeneity among the {name}s")
else:
    print(f"We Reject H0\nThere is Heterogeneity among the {name}s")

