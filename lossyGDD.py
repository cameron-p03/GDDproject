import csv
import struct
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf

# Read in data from csv file
def readNums(filename):
    file = csv.reader(open(filename, 'r'))
    numsIn = []
    min = 0
    max = 0
    for row in file:
        try:
            num = float(row[0])
            numsIn.append(num)
            if num<min or min==0:
                min = num
            if num>max or max==0:
                max = num
        except ValueError:
            print('error')
            continue
    return numsIn, max, min

def convertFloatToBin(floatIn):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floatIn))

#convert binary to floating point number
def convertBinToFloat(binIn):
    return struct.unpack('!f',struct.pack('!I', int(binIn, 2)))[0]

#decompress compressed values
def decompress(compressedNum, x, lineSegEquationDict, numBaseBitsRequired):
    base_bits = compressedNum[:numBaseBitsRequired]
    deviation = compressedNum[numBaseBitsRequired:]
    m = lineSegEquationDict[base_bits][0]
    c = lineSegEquationDict[base_bits][1]

    calculatedBase = (m*x)+c
    decompressedNum = float(calculatedBase) + convertBinToFloat(deviation + '0000000000000000000000')
    return decompressedNum

#
# Start of lossy compression incorporation
#
def compressAndFindRatio(breakNum):
    xVals = []
    yVals = []

    # Graph data value against its position in the list of data points
    numsIn, max, min = readNums("randomNumbers2.csv")
    for n in range(0, len(numsIn)):
        xVals.append(n+1)
        yVals.append(numsIn[n])

    plt.figure(1)
    plt.plot(xVals, yVals)
    plt.title('Value of data point Vs Order in data sequence')
    plt.xlabel('Order number of data point within sequence')
    plt.ylabel('Value of data point')


    x = np.array(xVals)
    y = np.array(yVals)


    # Calculate breakpoints (1 break for every 20 data points)
    numOfBreaks = breakNum
    print("NUM OF BREAKPOINTS: " + str(numOfBreaks))

    # Find number of bits required to store all bases (round up to nearest whole number)
    if(numOfBreaks < 2):
        numBaseBitsRequired = 1
    else:
        numBaseBitsRequired = math.ceil(math.log2(numOfBreaks))

    # Initialise PWLF using given data
    myPwlf = pwlf.PiecewiseLinFit(x,y)
    # Fit the data with specified breakpoints (every 10 data items)
    myPwlf.fit(numOfBreaks)

    # Predict for determined points
    xHat = np.linspace(1, len(numsIn), num=len(numsIn))
    yHat = myPwlf.predict(xHat)

    # print(str(xHat))
    # print(str(yHat))

    # Plot the results
    plt.figure(2)
    plt.plot(x, y, 'o', label='Data point')
    plt.plot(xHat, yHat, '-', label='Average using Piecewise Linear Regression')
    plt.title('Value of data point Vs Order in data sequence')
    plt.xlabel('Order number of data point within sequence')
    plt.ylabel('Value of data point')
    plt.legend(loc="upper left")
    plt.show()

    # Find deviation of each data point from the PWLF line
    deviationDict = {}

    pwlf_slopes = myPwlf.slopes #Gradient of each segment
    pwlf_breakpoints = myPwlf.fit_breaks.round() #Segment breakpoints rounded to nearest integer

    for x in range(0, len(numsIn)):
        deviation=float(numsIn[x]-yHat[x]) # Find deviation of real input value against predicted for each input number
        deviation = convertFloatToBin(deviation)[:10] # Take 10 most significant bits (lossy compression)
        deviationDict[x] = deviation # Store deviation
        

    # Store start point of each line segment, along with gradient and +c constant
    lineSegEquationDict = {}

    brkPntCtr = 0 # Track which line segment we're calculating for

    for breakpoint in pwlf_breakpoints:
        if (int(breakpoint) != len(numsIn)): # Can ignore final point as it is just the end point
            y = yHat[int(breakpoint)-1] # The line equations generated refer to the estimated y values, not the real input values, hence we use the estimates
            m = pwlf_slopes[brkPntCtr] # Assign the relevant gradient value
            x = int(breakpoint) # X value we're calculating from (breakpoint of line segment)
            c = y-(m*x) # Calculate constant C
            binaryDictKey = format(brkPntCtr, '0'+str(numBaseBitsRequired)+'b')  # Ensure binary representation has fixed length
            lineSegEquationDict[binaryDictKey] = (float(m),float(c)) # Store m and c in line segment dictionary
            brkPntCtr+=1

    # Test
    # print("LINE SEG EQUATION DICTIONARY (m,c): ")
    # print(lineSegEquationDict)


    # print("Actual value at x=8: " + str(numsIn[7]))
    # print("Predicted (PWLF) value at x=8: " + str(yHat[7]))
    # print("Deviation (truncated to most significant 16 bits): " + str(deviationDict[7]))
    # print("Deviation as float: " + str(convertBinToFloat(deviationDict[7]+'0000000000000000000000')))
    # print("Calculated value at x=8: " + str(((lineSegEquationDict['000'][0]*8)+lineSegEquationDict['000'][1] + convertBinToFloat(deviationDict[7]+'0000000000000000000000')))) # y = mx + c


    # Minimise number of bits used to represent base and deviation, then combine bit strings into one
    finalVals = [] # Store final values of base+deviation
    currentBreakpoint=0 
    nextBreakpoint = pwlf_breakpoints[1]

    for x in range(1, len(numsIn)+1):
        if(x == nextBreakpoint):
            currentBreakpoint+=1
            if (int(pwlf_breakpoints[currentBreakpoint+1]) != len(numsIn)): # Make sure next breakpoint isn't the final point
                nextBreakpoint = pwlf_breakpoints[currentBreakpoint+1]

        baseRefBinary = format(currentBreakpoint, '0'+str(numBaseBitsRequired)+'b')
        #print("BASE: " + str(baseRefBinary))
        deviation = deviationDict[x-1]
        #print("DEVIATION: " + str(deviation))
        finalVals.append(baseRefBinary+deviation)
        #print("finalVal: " + str(finalVals[x-1]))

    decompressedNums = []
    for x in range(1, len(numsIn)+1):
        decompressedNums.append(decompress(finalVals[x-1], x, lineSegEquationDict, numBaseBitsRequired))

    # Calculate compression ratio
    original_size = sum(len(convertFloatToBin(num)) for num in numsIn) # Calculate total number of bits in input
    compressed_size = ((sum(len(num) for num in finalVals)) + ((numBaseBitsRequired+64+64)*len(lineSegEquationDict))) # Size of all required decompression info
    compression_ratio = original_size / compressed_size
    print("COMPRESSION RATIO: " + str(compression_ratio))

    #  Find mean squared error
    n = len(numsIn)
    totalError=0
    for i in range(0, n):
        observedVal = numsIn[i]
        compressedVal = decompressedNums[i]
        totalError += ((observedVal-compressedVal)**2)

    meanError = totalError/n
    print("MEAN SQUARED ERROR: " + str(meanError))
    print()

for i in range(2,8):
    compressAndFindRatio(i)