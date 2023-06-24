#!/usr/bin/env python
# coding: utf-8

# ## Functions, Import Statements:

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import scipy.stats as ss
import pandas as pd

# To find the f critical value
#ss.f.ppf(q=1-0.05,df=4)

# To find the T critical value
#ss.t.ppf(q=1 - 0.05/2,df=4)


# ## WEEK 1:
# #### Write a python program to find the best fit straight line of the form y = a+bx and draw the scatter plot.

# In[2]:


# Linear Regression:

import matplotlib.pyplot as plt
import numpy as np

def mymean(data):
    Sum = 0
    n = len(data)
    for i in data:
        Sum += i
    return Sum/n

def drawPlotLine(x,y,Y_cal):
    Y_color = 'green'
    Y_cal_color = 'red'
    plt.plot(x,y,color=Y_color,marker='o',markerfacecolor='w')
    plt.plot(x,Y_cal,color=Y_cal_color,marker='o',markerfacecolor='w')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Best Fit Line")
    plt.show()
    print("X-Y Graph: Green")
    print("X-Ycal Graph: Red")

def LinearRegression(x,y):
    n = len(x)
    m = len(y)
    
    if n != m:
        print("Invalid Data!")
        return
    
    SumX = sum(x)
    SumY = sum(y)
    
    slope = (n*sum(x*y) - SumX*SumY) / (n*sum(x**2) - SumX**2)
    constant = (SumY - slope*SumX) / n
    
    Y_cal = slope*x + constant
    
    meanY = mymean(y)
    
    SST = sum((y-meanY)**2)
    SSE = sum((y-Y_cal)**2)
    SSR = sum((Y_cal-meanY)**2)
    
    Rsq = SSR/SST
    # Rsq = 1 - SSE/SST
    
    print("Y Calculated values: ",Y_cal)
    print("The Linear Regression Line for the given data:")
    print("Y = ({:.4f}) + ({:.4f})X".format(constant,slope))
    print("Sum of Squares due to Total: {:.4f}".format(SST))
    print("Sum of Squares due to Regression: {:.4f}".format(SSR))
    print("Sum of Squares due to Error: {:.4f}".format(SSE))
    print("R Square value: {:.4f}".format(Rsq))
    
    drawPlotLine(x,y,Y_cal)
    
    if Rsq > 0.9:
        print("The Regression Line Y = ({:.4f}) + ({:.4f})X is the best fit for given data".format(constant,slope))
    else:
        print("The Regression Line Y = ({:.4f}) + ({:.4f})X is not the best fit for given data".format(constant,slope))
        
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

LinearRegression(x,y)


# ## WEEK 2:
# #### Write a python program to fit a second degree parabola of the form y = a+bx+cx2 and draw the scatter plot.

# In[3]:


# Parabola Regression:

import matplotlib.pyplot as plt
import numpy as np

def mymean(data):
    Sum = 0
    n = len(data)
    for i in data:
        Sum += i
    return Sum/n

def drawPlotParabola(x,y,Y_cal):
    Y_color = 'green'
    Y_cal_color = 'red'
    plt.plot(x,y,color=Y_color,marker='o',markerfacecolor='w')
    plt.plot(x,Y_cal,color=Y_cal_color,marker='o',markerfacecolor='w')
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title("Best Fit Curve")
    plt.show()
    print("X-Y Graph: Green")
    print("X-Ycal Graph: Red")

def DET(matrix):
    Sum = 0
    
    Sum += matrix[0][0] * (matrix[1][1]*matrix[2][2] - matrix[2][1]*matrix[1][2])
    Sum -= matrix[0][1] * (matrix[1][0]*matrix[2][2] - matrix[2][0]*matrix[1][2])
    Sum += matrix[0][2] * (matrix[1][0]*matrix[2][1] - matrix[2][0]*matrix[1][1])
    
    return Sum
    
def ParabolaRegression(x,y):
    n = len(x)
    m = len(y)

    if n != m:
        print("Invalid Data!")
        return
    
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    xy = x*y
    x2y = x2*y
    
    Sumx = sum(x)
    Sumx2 = sum(x2)
    Sumx3 = sum(x3)
    Sumx4 = sum(x4)
    Sumy = sum(y)
    Sumxy = sum(xy)
    Sumx2y = sum(x2y)
    
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
    
    Y_cal = a + b*x + c*x2
    
    mean_Y = mymean(y)
    
    SST = sum((y-mean_Y)**2)
    SSE = sum((y-Y_cal)**2)
    SSR = sum((Y_cal-mean_Y)**2)
    
    Rsq = SSR/SST
    # Rsq = 1 - SSE/SST
    
    print("Y Calculated values: ",Y_cal)
    print("The Parabola Regression Line for the given data: ")
    print("Y = ({:.4f}) + ({:.4f})X + ({:.4f})X2".format(a,b,c))
    print("Sum of Squares due to Total: {:.4f}".format(SST))
    print("Sum of Squares due to Regression: {:.4f}".format(SSR))
    print("Sum of Squares due to Error: {:.4f}".format(SSE))
    print("R Square value: {:.4f}".format(Rsq))
    
    drawPlotParabola(x,y,Y_cal)
    
    if Rsq > 0.9:
        print("The Regression Curve Y = ({:.4f}) + ({:.4f})X + ({:.4f})X2 is the best fit for given data.".format(a,b,c))
    else:
        print("The Regression Curve Y = ({:.4f}) + ({:.4f})X + ({:.4f})X2 is not the best fit for given data.".format(a,b,c))

x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

ParabolaRegression(x, y)


# ## WEEK 3:
# #### Write a python program to find Karl Pearson’s correlation coefficient between X and Y variables.

# In[5]:


# Karl Pearson's Correlation Coefficient:

import numpy as np
from math import sqrt

def mymean(data):
    Sum = 0
    n = len(data)
    for i in data:
        Sum += i
        
    return Sum/n

def CovXY(x,y):
    n = len(x)
    meanX = mymean(x)
    meanY = mymean(y)
    SumXY = sum(x*y)
    CovXY = (SumXY/n) - (meanX*meanY)
    
    return CovXY
    
def SD(x):
    n = len(x)
    meanx = mymean(x)
    Sumx2 = sum(x**2)
    var = (Sumx2/n) - (meanx**2)
    
    return sqrt(var)

def KPCC(x,y):
    if len(x) != len(y):
        print("Invalid Data!")
        return
    
    Covxy = CovXY(x,y)
    SDx = SD(x)
    SDy = SD(y)
    
    KPCC = Covxy / (SDx*SDy)
    
    print("Co-variance of X and Y = {:.4f}".format(Covxy))
    print("Standard Deviation of X = {:.4f}".format(SDx))
    print("Standard Deviation of Y = {:.4f}".format(SDy))
    
    if KPCC > -1 and KPCC < 1:
        print("The Karl Pearson's Correlation Coefficient of the given data is: {:.4f}".format(KPCC))
    else:
        print("Invalid Data!")
    
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

KPCC = KPCC(x,y)


# ## WEEK 4:
# #### Write a python program to find the Spearman’s correlation coefficient between X and Y variables.

# In[4]:


# Spearman's Rank Correlation Coefficient:

import pandas as pd
import numpy as np

def Rankify(data):
    N = len(data)
    Ranks = [None for i in range(N)]
    
    for i in range(N):
        BigNums = 1
        SameNums = 0
        
        # Count no of bigger elements from 0 to current number-1
        for j in range(i):
            if (data[j] > data[i]):
                BigNums += 1
            if (data[j] == data[i]):
                SameNums += 1
                
        # Count no of bigger elements from current number+1 to N-1
        for j in range(i+1,N):
            if (data[j] > data[i]):
                BigNums += 1
            if (data[j] == data[i]):
                SameNums += 1
                
        # Use Fractional Rank formula
        # fractional_rank = BigNums + SameNums/2
        Ranks[i] = BigNums + SameNums/2

    return Ranks

def RemDups(data):
    set_data = []
    
    for i in data:
        if i not in set_data:
            set_data.append(i)
            
    return set_data

def CF(x,y):
    cf = 0
    x = list(x)
    y = list(y)
    set_x = RemDups(x)
    set_y = RemDups(y)
    
    for i in set_x:
        count_x = x.count(i)
        if count_x > 1:
            cf += (count_x * (count_x**2 - 1)) / 12
            
    for i in set_y:
        count_y = y.count(i)
        if count_y > 1:
            cf += (count_y * (count_y**2 - 1)) / 12
    
    return cf

def SRCC(x,y):
    n = len(x)
    m = len(y)

    if n != m:
        print("Invalid Data!")
        return
    
    RankX = np.array(Rankify(x))
    RankY = np.array(Rankify(y))
    
    # Difference of Ranks:
    Di = RankX - RankY
    DiSq = Di**2
    
    SumDiSq = sum(DiSq)
    
    # Correction Factor:
    cf = CF(x,y)
    SumDiSq += cf

    SRCC = 1 - ((6 * SumDiSq) / (n * (n**2 - 1)))
    
    SRCC_Table = {
        "X values": x,
        "Y values": y,
        "Ranks of X": RankX,
        "Ranks of Y": RankY,
        "Di values": Di,
        "Di sq. values": DiSq}
    
    print(pd.DataFrame(SRCC_Table))
    
    print("Correction Factor: ",cf)
    print("Sum of Di sq: ",SumDiSq)
    print("The Spearman's Ranked Correlation Coefficient of the given data: {:.4f}".format(SRCC))
    
x = np.array([float(i) for i in input("Enter x values: ").strip().split()])
y = np.array([float(i) for i in input("Enter y values: ").strip().split()])

SRCC(x,y)


# ## WEEK 5:
# #### Write a python program to classify the data based on one way ANOVA.

# In[12]:


# ANOVA One Way Classification:

import pandas as pd
import numpy as np
import scipy.stats as ss

k = int(input("Enter the number of Treatments: "))
#k = 4
Treatment = input("Enter name of the Treatments: ")
#Treatment = "Technician"

Data = []

#Data = [[6, 14, 10 ,8 ,11], [14, 9 ,12, 10, 14], [10 ,12 ,7, 15, 11],[9, 12, 8, 10, 11]]

for i in range(k):
    a = np.array([float(j) for j in input(f"Enter {Treatment} {i+1} values: ").strip().split()])
    Data.append(a)

Data = np.array(Data)
alpha = float(input("Enter Level of Significance: "))
#alpha = 0.01

print(f"\nNull Hypothesis (H0): There is Homogenity among the {Treatment}s")
print(f"Alternate Hypothesis (H1): There is Heterogenity among the {Treatment}s\n")

# Total No. of values (N):
# Sum of Treatments (Ti):
# Row sum of squares (RSS):
N = Ti = Ti2 = RSS = 0
for i in Data:
    Ti += sum(i)
    Ti2 += sum(i)**2 / len(i)
    RSS += sum(i**2)
    N += len(i)

# Correction Factor (CF):
CF = Ti**2 / N

# Sum of Squares due to Total (SST):
SST = RSS - CF
# Sum of Squares due to Treatments (SSTr):
SSTr = Ti2 - CF
# Sum of Squares due to Error (SSE):
SSE = SST - SSTr

# Mean Sum of Squares due to Treatments (MeanSSTr):
MeanSSTr = SSTr/(k-1)
# Mean Sum of Squares due to Error (MeanSSE):
MeanSSE = SSE/(N-k)

# F calculated value:
Fcal = MeanSSTr / MeanSSE

# F table value:
FTable = ss.f.ppf(1-alpha, k-1, N-k)

# Degrees of Freedom:
DOF = f"~ F({k-1},{N-k})"

if(Fcal < 1):
    Fcal = 1 / Fcal
    FTable = ss.f.ppf(1-alpha, N-k, k-1)
    DOF = f"~ F({N-k},{k-1})"

print("Sum of Squares due to Total (SST): {:.4f}".format(SST))
print("Sum of Squares due to Treatments (SSTr): {:.4f}".format(SSTr))
print("Sum of Squares due to Error (SSE): {:.4f}\n".format(SSE))

print("Mean Sum of Squares due to Treatments (Mean SST): {:.4f}".format(MeanSSTr))
print("Mean Sum of Squares due to Error (Mean SSE): {:.4f}\n".format(MeanSSE))

ANOVA_One_Way_Classification_Table = {
    "S O V": [Treatment, "Error", "Total"],
    "S O S": ["{:.4f}".format(SSTr), "{:.4f}".format(SSE), "{:.4f}".format(SST)],
    "D O F": [k-1, N-k, N-1],
    "M S O S": ["{:.4f}".format(MeanSSTr), "{:.4f}".format(MeanSSE), " - "],
    "V R": ["F-cal = {:.4f}".format(Fcal), DOF, ""]
}

data_frame = pd.DataFrame(ANOVA_One_Way_Classification_Table)

print(data_frame) 

print(f"\nF-Calculated Value: {Fcal}")
print(f"F-Table Value: {FTable}\n")

if(Fcal < FTable):
    print(f"We Accept H0\nThere is Homogeneity among the {Treatment}s")
else:
    print(f"We Reject H0\nThere is Heterogeneity among the {Treatment}s")


# ## WEEK 6:
# #### Write a python program to classify the data based on two way ANOVA.

# In[11]:


# ANOVA Two Way Classification:

import pandas as pd
import numpy as np
import scipy.stats as ss

def mymean(data):
    Sum = 0
    n = len(data)
    for i in data:
        Sum += i
        
    return Sum/n

k = int(input("Enter the number of Treatments: "))
h = int(input("Enter the number of Blocks: "))
#k = 4
#h = 5

Treatment = input("Enter name of the Treatments: ")
Block = input("Enter name of the Blocks: ")
#Treatment = "Form"
#Block = "Student"

Data = []
for i in range(k):
    row = []
    while len(row) != h:
        row = np.array([float(i) for i in input(f"Enter {Treatment} {i+1} values: ").strip().split()])
    Data.append(row)
                       
Data = np.array(Data)

alpha = float(input("Enter Level of Significance: "))
#alpha = 0.05

print(f"\nNull Hypothesis: There is Homogenity among the {Treatment}s")
print(f"Alternate Hypothesis: There is Heterogenity among the {Treatment}s\n")

# Total no. of values (N):
N = k * h

# Sum of Treatments (Ti):
Ti = np.array([sum(i) for i in Data])
Ti2 = Ti**2
SumTi2 = sum(Ti2)

# Sum of Blocks (Bj):
Bj = Data[0]
for i in range(1,k):
    Bj = Bj + Data[i]
    
Bj2 = Bj**2
SumBj2 = sum(Bj2)

# Grand Total (G):
G = sum(Ti) 

# Row Sum of Squares (RSS):
RSS = 0
for i in Data:
    for j in i:
        RSS += j**2
        
# Correction Factor (CF):
CF = G**2 / N

# Sum of Squares due to Total (SST):
SST = RSS - CF
# Sum of Squares due to Treatments (SSTr):
SSTr = SumTi2/h - CF
# Sum of Squares due to Blocks (SSB):
SSB = SumBj2/k - CF
# Sum of Squares due to Error (SSE):
SSE = SST - SSTr - SSB

# Mean Sum of Squares due to Treatments (MeanSSTr):
MeanSSTr = SSTr/(k-1)
# Mean Sum of Squares due to Blocks (MeanSSB):
MeanSSB = SSB/(h-1)
# Mean Sum of Squares due to Error (MeanSSE):
MeanSSE = SSE/((k-1)*(h-1))

# F calculated value of Treatments (F(Tr)):
F_Tr_cal = MeanSSTr / MeanSSE
# F calculated value of Blocks (F(B)):
F_B_cal = MeanSSB / MeanSSE

# F table value of Treatments:
F_Tr_Table = ss.f.ppf(1-alpha, k-1, (k-1)*(h-1))
# F table value of Blocks:
F_B_Table = ss.f.ppf(1-alpha, h-1, (k-1)*(h-1))

# Degrees of Freedom for Treatments:
F_Tr_DOF = f"~ F({k-1},{(k-1)*(h-1)})"
# Degrees of Freedom for Blocks:
F_B_DOF = f"~ F({h-1},{(k-1)*(h-1)})"

if(F_Tr_cal < 1):
    F_Tr_cal = 1 / F_Tr_cal
    F_Tr_Table = ss.f.ppf(1-alpha,(k-1)*(h-1), k-1)
    F_Tr_DOF = f"~ F({(k-1)*(h-1)},{k-1})"

if(F_B_cal < 1):
    F_B_cal = 1 / F_B_cal
    F_B_Table = ss.f.ppf(1-alpha,(k-1)*(h-1), h-1)
    F_B_DOF = f"~ F({(k-1)*(h-1)},{h-1})"

print("Sum of Squares due to Total (SST): {:.4f}".format(SST))
print("Sum of Squares due to Treatments (SSTr): {:.4f}".format(SSTr))
print("Sum of Squares due to Blocks (SSB): {:.4f}".format(SSB))
print("Sum of Squares due to Error (SSE): {:.4f}\n".format(SSE))

print("Mean Sum of Squares due to Treatments (Mean SSTr): {:.4f}".format(MeanSSTr))
print("Mean Sum of Squares due to Blocks (Mean SSB): {:.4f}".format(MeanSSB))
print("Mean Sum of Squares due to Error (Mean SSE): {:.4f}\n".format(MeanSSE))

ANOVA_Two_Way_Classification_Table = {
    "S O V": [Treatment, Block, "Error", "Total"],
    "S O S": ["{:.4f}".format(SSTr), "{:.4f}".format(SSB), "{:.4f}".format(SSE), "{:.4f}".format(SST)],
    "D O F": [k-1, h-1, (k-1)*(h-1), (k*h)-1],
    "M S O S": ["{:.4f}".format(MeanSSTr), "{:.4f}".format(MeanSSB), "{:.4f}".format(MeanSSE), " - "],
    "V R": ["F(Tr)-cal = {:.4f}".format(F_Tr_cal), F_Tr_DOF, 
            "F(B)-cal = {:.4f}".format(F_B_cal), F_B_DOF]}

data_frame = pd.DataFrame(ANOVA_Two_Way_Classification_Table)

print(data_frame) 

print(f"\nInference Related to {Treatment}s:")
print("F(Tr)-Calculated Value: {:.4f}".format(F_Tr_cal))
print("F(Tr)-Table Value: {:.4f}\n".format(F_Tr_Table))

if(F_Tr_cal < F_Tr_Table):
    print(f"We Accept H0(Tr)\nThere is Homogeneity among the {Treatment}s\n")
else:
    print(f"We Reject H0(Tr)\nThere is Heterogeneity among the {Treatment}s\n")
    
print(f"\nInference Related to {Block}s:")
print("F(B)-Calculated Value: {:.4f}".format(F_B_cal))
print("F(B)-Table Value: {:.4f}\n".format(F_B_Table))

if(F_B_cal < F_B_Table):
    print(f"We Accept H0(B)\nThere is Homogeneity among the {Block}s\n")
else:
    print(f"We Reject H0(B)\nThere is Heterogeneity among the {Block}s\n")


# ## WEEK 7:
# #### Write a python program to fit a multiple regression model for any given data.

# In[12]:


# Multiple Linear Regression Model:

def GoodnessOfFit(y , Yfitted):
    print("To Test Goodness of Fit using Determination of Coefficients (R2):")
    Residual = y - Yfitted
    y_ybar = y-ybar

    SSE = mysum(Residual**2)
    SST = mysum(y_ybar**2)
    SSR = SST - SSE

    Rsq = SSR/SST

    print(f"Y fitted values: {Yfitted}")
    print(f"Residual values: {Residual}")
    print(f"Y - Ybar values: {y_ybar}")
    print(f"Sum of Squares due to Error (SSE): {SSE}")
    print(f"Sum of Squares due to Total (SST): {SST}")
    print(f"Sum of Squares due to Regression (SSR): {SSR}")
    print(f"Coefficient of Determination (R2): {Rsq}")

    if Rsq < 0.9:
        print("The Multiple Regression Model is not a Good Fit for the given data.")
    else:
        print("The Multiple Regression Model is a Good Fit for the given data.")
        
def ParameterTesting(BetaCap, XTX_inverse):
    print("To Test the individual parameters using T-Test:")
    
        
# variables = int(input("Enter no. of variables in the model:"))
variables = 2

if variables == 2:
    #y = np.array([float(i) for i in input("Enter y values: ").strip().split()])
    #x1 = np.array([float(i) for i in input("Enter x1 values: ").strip().split()])
    y = np.array([10,20,30,40,50])
    x1 = np.array([5,7,10,12,20])
    
    leny = len(y)
    lenx1 = len(x1)
    
    if lenx1 != leny:
        print("Invalid Data!")
    else:
        x0 = np.array([1 for i in range(lenx1)])
        X = [x0,x1]
        
        XTX = np.array([[mysum(x0**2),mysum(x0*x1)],
                        [mysum(x1*x0),mysum(x1**2)]])
        
        XTX_inverse = np.linalg.inv(XTX)
        
        XTY = np.array([mysum(x0*y), mysum(x1*y)])
        
        BetaCap = [mysum(XTX_inverse[0] * XTY), mysum(XTX_inverse[1] * XTY)]
        
        Yfitted = np.array(BetaCap[0] + BetaCap[1]*x1)
        print("Multiple Linear Regression Model for the given data:")
        print(f"Y = {BetaCap[0]} + {BetaCap[1]}X1")
        
        GoodnessOfFit(y,Yfitted)
        
if variables == 3:
    #y = np.array([float(i) for i in input("Enter y values: ").strip().split()])
    #x1 = np.array([float(i) for i in input("Enter x1 values: ").strip().split()])
    #x2 = np.array([float(i) for i in input("Enter x2 values: ").strip().split()])
    y = np.array([100,110,105,94,95,99,104,108,105,98,103,110])
    x1 = np.array([9,8,7,14,12,10,7,4,6,5,7,6])
    x2 = np.array([62,58,64,60,63,57,55,56,59,61,57,60])
    
    leny = len(y)
    lenx1 = len(x1)
    lenx2 = len(x2)
    ybar = mymean(y)

    if leny != lenx1 or leny != lenx2 or lenx1 != lenx2:
        print("Invalid Data!")
    else:
        x0 = np.array([1 for i in range(lenx1)])
        X = [x0,x1,x2]
        
        XTX = np.array([[mysum(x0**2),mysum(x0*x1),mysum(x0*x2)],
                        [mysum(x1*x0),mysum(x1**2),mysum(x1*x2)],
                        [mysum(x2*x0),mysum(x2*x1),mysum(x2**2)]])
        
        XTX_inverse = np.linalg.inv(XTX)
        
        XTY = np.array([mysum(x0*y), mysum(x1*y),mysum(x2*y)])
        
        BetaCap = [mysum(XTX_inverse[0] * XTY), mysum(XTX_inverse[1] * XTY),mysum(XTX_inverse[2] * XTY)]
        
        Yfitted = np.array(BetaCap[0] + BetaCap[1]*x1 + BetaCap[2]*x2)
        print("Multiple Linear Regression Model for the given data:")
        print(f"Y = {BetaCap[0]} + {BetaCap[1]}X1 + {BetaCap[2]}X2")
        
        GoodnessOfFit(y,Yfitted)
        
        ParameterTesting(BetaCap, XTX_inverse)
        
else:
    print("This program cannot find MLR Models for more than 3 variable data.")


# ## WEEK 8:
# #### Write a python program to fit a multivariate regression model for any given data.

# ## WEEK 9:
# #### Write a python program to classify the treatments based on MANOVA Test.

# ## WEEK 10:
# #### Write a python program to classify the given observations using Linear Discriminant Analysis.

# ## WEEK 11:
# #### Write a python program to find Principle components for the given variables.

# ## WEEK 12:
# #### Write a python program to group the given variables using Factor Analysis.
