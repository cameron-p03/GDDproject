import csv
import struct
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf
import time

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
def decompress(compressedNum, x, lineSegEquationDict, numBaseBitsRequired, numTruncatedBits, returnBase):
    base_bits = compressedNum[:numBaseBitsRequired]
    deviation = compressedNum[numBaseBitsRequired:]
    m = lineSegEquationDict[base_bits][0]
    c = lineSegEquationDict[base_bits][1]

    calculatedBase = (m*x)+c
    decompressedNum = float(calculatedBase) + convertBinToFloat(deviation + ('0' * (numTruncatedBits)))
    
    if returnBase:
        return float(calculatedBase)
    else:
        return decompressedNum

#
# Start of lossy compression incorporation
#
def compressAndFindRatio(breakNum, lossy):
    start = time.time()
    xVals = []
    yVals = []

    # Graph data value against its position in the list of data points
    numsIn, max, min = readNums("randomNumbers.csv")
    for n in range(0, len(numsIn)):
        xVals.append(n+1)
        yVals.append(numsIn[n])
    
    # plt.figure(1)
    # plt.plot(xVals, yVals)
    # plt.title('Value of data point Vs Order in data sequence')
    # plt.xlabel('Order number of data point within sequence')
    # plt.ylabel('Value of data point')


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
    # Fit the data with specified breakpoints
    myPwlf.fit(numOfBreaks)

    # Predict for determined points
    xHat = np.linspace(1, len(numsIn), num=len(numsIn))
    yHat = myPwlf.predict(xHat)

    # print(str(xHat))
    # print(str(yHat))

    # Plot the results
    # plt.figure(2)
    # plt.plot(x, y, 'o', label='Data point')
    # plt.plot(xHat, yHat, '-', label='Average using Piecewise Linear Regression')
    # plt.title('Value of data point Vs Order in data sequence')
    # plt.xlabel('Order number of data point within sequence')
    # plt.ylabel('Value of data point')
    # plt.legend(loc="upper left")
    #plt.show()

    # Find deviation of each data point from the PWLF line
    deviationDict = {}

    pwlf_slopes = myPwlf.slopes #Gradient of each segment
    pwlf_breakpoints = myPwlf.fit_breaks.round() #Segment breakpoints rounded to nearest integer

    for x in range(0, len(numsIn)):
        deviation=float(numsIn[x]-yHat[x]) # Find deviation of real input value against predicted for each input number
        deviation = convertFloatToBin(deviation)
        if lossy:
            deviation = deviation[:-lossy] # Take most significant bits (lossy compression)
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
        #print(baseRefBinary + deviation)
        #print("finalVal: " + str(finalVals[x-1]))

    decompressedNums = []
    for x in range(1, len(numsIn)+1):
        decompressedNums.append(decompress(finalVals[x-1], x, lineSegEquationDict, numBaseBitsRequired, lossy, 0))

    with open('decompressedVals.csv', 'w') as file:
        csvwriter = csv.writer(file, delimiter='\n')
        csvwriter.writerow(decompressedNums)

    # Calculate compression ratio
    original_size = sum(len(convertFloatToBin(num)) for num in numsIn) # Calculate total number of bits in input
    compressed_size = ((sum(len(num) for num in finalVals)) + ((numBaseBitsRequired+64+64)*len(lineSegEquationDict))) # Size of all required decompression info
    compressionRatio = original_size / compressed_size
    print("COMPRESSION RATIO: " + str(compressionRatio))

    #  Find mean squared error
    n = len(numsIn)
    totalError=0
    for i in range(0, n):
        observedVal = numsIn[i]
        compressedVal = decompressedNums[i]
        totalError += abs(observedVal-compressedVal)

    meanError = totalError/n

    # Find mean squared error FROM ONLY BASES
    n = len(numsIn)
    totalBaseError=0
    for i in range(1, n+1):
        observedVal = float(numsIn[i-1])
        baseVal = decompress(finalVals[i-1], i, lineSegEquationDict, numBaseBitsRequired, lossy, 1)
        totalBaseError += abs(observedVal-baseVal)

    meanBaseError = totalBaseError/n

    end = time.time()
    elapsedTime = end-start

    print("REAL VAL: " + str(numsIn[-1])+ ". STORED DEVIATION: "+str(convertBinToFloat(deviationDict[len(numsIn)-1] + '0'*(lossy)))+ ". BASE VALUE: " + str(decompress(finalVals[-1], len(numsIn), lineSegEquationDict, numBaseBitsRequired, lossy, 1)) + ". DECOMPRESSED ESTIMATE: " +str(decompressedNums[-1]))
    print("MEAN ABSOLUTE ERROR WITH FULL DECOMPRESSION: " + str(meanError))
    print("MEAN ABSOLUTE ERROR FROM ONLY BASES: " + str(meanBaseError))
    print("EXECUTION TIME (s): " + str(elapsedTime))
    print()

    # xVals = []
    # yHat = []

    # Graph data value against its position in the list of data points
    # for n in range(0, len(numsIn)):
    #     xVals.append(n+1)
    #     yHat.append(decompress(finalVals[n], n+1, lineSegEquationDict, numBaseBitsRequired, lossy, 0))
    
    # x = np.array(xVals)
    # y = np.array(yVals)

    # plt.figure(3)
    # plt.plot(x, y, 'o', label='Data point')
    # plt.plot(xVals, yHat, '-', label='DECOMPRESSED ESTIMATE')
    # plt.title('Value of data point Vs Order in data sequence')
    # plt.xlabel('Order number of data point within sequence')
    # plt.ylabel('Value of data point')
    # plt.legend(loc="upper left")
    # plt.show()

    return compressionRatio, meanError, meanBaseError, elapsedTime

numberOfBreakPoints=[]
compressionRatioResults=[]
meanAverageErrorResults=[]
baseMeanAverageErrorResults=[]
timeResults=[]

truncateBits = int(input("Please enter number of bits to truncate from deviation values (higher num = more compression, max=32): "))

for i in range(2,8):
    results = compressAndFindRatio(i, truncateBits)
    numberOfBreakPoints.append(i)
    compressionRatioResults.append(results[0])
    meanAverageErrorResults.append(results[1])
    baseMeanAverageErrorResults.append(results[2])
    timeResults.append(results[3])



plt.figure(1)
plt.plot(numberOfBreakPoints, compressionRatioResults, '-', label='Compression Ratio')
plt.title('Number of Break Points Vs Compression Ratio Achieved')
plt.xlabel('Number of Break Points')
plt.ylabel('Compression Ratio')
plt.legend(loc="upper left")

plt.figure(2)
plt.plot(numberOfBreakPoints, meanAverageErrorResults, '-', label='Decompressed value MAE')
plt.plot(numberOfBreakPoints, baseMeanAverageErrorResults, '-', label='Base value MAE')
plt.title('Number of Break Points Vs Mean Average Errors recorded')
plt.xlabel('Number of Break Points')
plt.ylabel('Mean Average Error')
plt.legend(loc="upper left")

plt.figure(3)
plt.plot(numberOfBreakPoints, timeResults, '-', label='Execution Time (s)')
plt.title('Number of Break Points Vs Algorithm Execution Time')
plt.xlabel('Number of Break Points')
plt.ylabel('Execution Time (s)')
plt.legend(loc="upper left")
plt.show()