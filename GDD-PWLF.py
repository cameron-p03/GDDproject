import csv
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf
import time

# Read in data from csv file
def readNums(filename, rowIndexOfValueToRead, maxNums):
    rowNum=0
    file = csv.reader(open(filename, 'r'))
    numsIn = []
    for row in file:
        if rowNum <maxNums:
            try:
                num = float(row[rowIndexOfValueToRead])
                numsIn.append(num)
            except ValueError:
                print('error')
                continue
            rowNum+=1
        else:
            return numsIn
    return numsIn

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
def compressAndFindRatio(numsIn, segments, lossy, fast):
    compressStart = time.time()
    xVals = []
    yVals = []

    length=len(numsIn)

    # Graph data value against its position in the list of data points
    for n in range(0, length):
        xVals.append(n+1)
        yVals.append(numsIn[n])
    
    # plt.figure(1)
    # plt.plot(xVals, yVals)
    # plt.title('Value of data point Vs Order in data sequence')
    # plt.xlabel('Order number of data point within sequence')
    # plt.ylabel('Value of data point')

    x = np.array(xVals)
    y = np.array(yVals)

    # Find number of bits required to store all bases (round up to nearest whole number)
    if(segments < 2):
        numBaseBitsRequired = 1
    else:
        numBaseBitsRequired = math.ceil(math.log2(segments))

    # Initialise PWLF using given data
    myPwlf = pwlf.PiecewiseLinFit(x,y)
    # Fit the data with specified breakpoints
    if fast==True:
        myPwlf.fitfast(segments)
    else:
        myPwlf.fit(segments)

    # Predict for determined points
    yHat = myPwlf.predict(xVals)

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

    for x in range(0, length):
        deviation=float(numsIn[x]-yHat[x]) # Find deviation of real input value against predicted for each input number
        deviation = convertFloatToBin(deviation)
        if lossy:
            deviation = deviation[:-lossy] # Take most significant bits (lossy compression)
        deviationDict[x] = deviation # Store deviation
        

    # Store gradient and +c constant for each line segment
    lineSegEquationDict = {}

    brkPntCtr = 0 # Track which line segment we're calculating for

    for breakpoint in pwlf_breakpoints:
        if (int(breakpoint) != length): # Can ignore final point as it is just the end point
            y = yHat[int(breakpoint)-1] # The line equations generated refer to the estimated y values, not the real input values, hence we use the estimates
            m = pwlf_slopes[brkPntCtr] # Assign the relevant gradient value
            x = int(breakpoint) # X value we're calculating from (breakpoint of line segment)
            print(x)
            c = y-(m*x) # Calculate constant C
            binaryDictKey = format(brkPntCtr, '0'+str(numBaseBitsRequired)+'b')  # Ensure binary representation has fixed length
            lineSegEquationDict[binaryDictKey] = (float(m),float(c)) # Store m and c in line segment dictionary
            brkPntCtr+=1

    # Test
    #print(pwlf_breakpoints)
    #print("LINE SEG EQUATION DICTIONARY (m,c): ")
    #print(lineSegEquationDict)


    # print("Actual value at x=8: " + str(numsIn[7]))
    # print("Predicted (PWLF) value at x=8: " + str(yHat[7]))
    # print("Deviation (truncated to most significant 16 bits): " + str(deviationDict[7]))
    # print("Deviation as float: " + str(convertBinToFloat(deviationDict[7]+'0000000000000000000000')))
    # print("Calculated value at x=8: " + str(((lineSegEquationDict['000'][0]*8)+lineSegEquationDict['000'][1] + convertBinToFloat(deviationDict[7]+'0000000000000000000000')))) # y = mx + c


    # Minimise number of bits used to represent base and deviation, then combine bit strings into one
    finalVals = [] # Store final values of base+deviation
    currentBreakpoint=0 
    nextBreakpoint = pwlf_breakpoints[1]

    if segments !=1:
        for x in range(1, length+1):
            if(x == nextBreakpoint):
                currentBreakpoint+=1
                if (int(pwlf_breakpoints[currentBreakpoint+1]) != length): # Make sure next breakpoint isn't the final point
                    nextBreakpoint = pwlf_breakpoints[currentBreakpoint+1]

            baseRefBinary = format(currentBreakpoint, '0'+str(numBaseBitsRequired)+'b')
            #print("BASE: " + str(baseRefBinary))
            deviation = deviationDict[x-1]
            #print("DEVIATION: " + str(deviation))
            finalVals.append(baseRefBinary+deviation)
            #print(baseRefBinary + deviation)
            #print("finalVal: " + str(finalVals[x-1]))
    else:
        for x in range(0,length):
            finalVals.append('0' + deviationDict[x])

    compressEnd = time.time()

    decompressedNums = []
    decompressStart=time.time()
    for x in range(1, length+1):
        decompressedNums.append(decompress(finalVals[x-1], x, lineSegEquationDict, numBaseBitsRequired, lossy, 0))

    decompressEnd=time.time()

    # with open('decompressedVals.csv', 'w') as file:
    #     csvwriter = csv.writer(file, delimiter='\n')
    #     csvwriter.writerow(decompressedNums)

    # Calculate compression ratio
    original_size = sum(len(convertFloatToBin(num)) for num in numsIn) # Calculate total number of bits in input
    compressed_size = ((sum(len(num) for num in finalVals)) + ((numBaseBitsRequired+64+64)*len(lineSegEquationDict))) # Size of all required decompression info
    compressionRatio = original_size / compressed_size
    #print("COMPRESSION RATIO: " + str(compressionRatio))

    #  Find mean absolute error
    n = length
    totalError=0
    for i in range(0, n):
        observedVal = numsIn[i]
        compressedVal = decompressedNums[i]
        totalError += abs(observedVal-compressedVal)

    meanError = totalError/n

    # Find mean absolute error FROM ONLY BASES
    n = length
    totalBaseError=0
    for i in range(1, n+1):
        observedVal = float(numsIn[i-1])
        baseVal = decompress(finalVals[i-1], i, lineSegEquationDict, numBaseBitsRequired, lossy, 1)
        totalBaseError += abs(observedVal-baseVal)

    meanBaseError = totalBaseError/n

    compressTime = compressEnd-compressStart
    decompressTime = decompressEnd - decompressStart

    #print("REAL VAL: " + str(numsIn[-1])+ ". STORED DEVIATION: "+str(convertBinToFloat(deviationDict[length-1] + '0'*(lossy)))+ ". BASE VALUE: " + str(decompress(finalVals[-1], length, lineSegEquationDict, numBaseBitsRequired, lossy, 1)) + ". DECOMPRESSED ESTIMATE: " +str(decompressedNums[-1]))
    #print("BREAKPOINTS: "+str(segments)+". COMPRESSION RATIO: "+str(compressionRatio)+". MEAN ABSOLUTE ERROR WITH FULL DECOMPRESSION: " + str(meanError)+". MEAN ABSOLUTE ERROR FROM ONLY BASES: " + str(meanBaseError)+". EXECUTION TIME (s): " + str(elapsedTime))

    # xVals = []
    # yHat = []

    # Graph data value against its position in the list of data points
    # for n in range(0, length):
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

    return compressionRatio, meanError, meanBaseError, compressTime, decompressTime


maxNums = 1425
numsIn = readNums("synthetic_1.csv", 1, maxNums)
# Used to hold the results from algorithm execution (for graphing)
compressionRatioResults = [[]*1 for i in range(8)]
MeanAbsoluteErrorResults=[[]*1 for i in range(8)]
baseMeanAbsoluteErrorResults=[[]*1 for i in range(8)]
compressTimeResults=[[]*1 for i in range(8)]
decompressTimeResults=[[]*1 for i in range(8)]
numberOfBreakPoints=[[]*1 for i in range(8)]

#truncateBits = int(input("Please enter number of bits to truncate from deviation values (higher num = more compression, max=32): "))

resultCtr=0
fast=True
setting=1

if setting==1:
    for truncateBits in range(0, 29, 4):
        for breakpoints in range(0, 7):
            results = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)
            results2 = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)  # I run the function 3 times and record the mean to increase reliability of the data obtained
            results3 = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)

            numberOfBreakPoints[resultCtr].append(breakpoints)
            compressionRatioResults[resultCtr].append((results[0]+results2[0]+results3[0])/3)
            MeanAbsoluteErrorResults[resultCtr].append((results[1]+results2[1]+results3[1])/3)
            baseMeanAbsoluteErrorResults[resultCtr].append((results[2]+results2[2]+results3[2])/3)
            compressTimeResults[resultCtr].append((results[3]+results2[3]+results3[3])/3)
            decompressTimeResults[resultCtr].append((results[4]+results2[4]+results3[4])/3)

            #print("TRUNCATED BITS: "+str(truncateBits)+". BREAKPOINTS: "+str(breakpoints)+". COMPRESSION RATIO: "+str(compressionRatioResults[resultCtr][-1])+". DECOMPRESSED MAE: " +str(MeanAbsoluteErrorResults[resultCtr][-1])+". BASE MAE: "+str(baseMeanAbsoluteErrorResults[resultCtr][-1])+". EXECUTION TIME: "+str(timeResults[resultCtr][-1]))

        resultCtr+=1


    # GRAPH OBTAINED RESULTS
    plt.figure(1)
    plt.plot(numberOfBreakPoints[0], compressionRatioResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBreakPoints[1], compressionRatioResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBreakPoints[2], compressionRatioResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBreakPoints[3], compressionRatioResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBreakPoints[4], compressionRatioResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBreakPoints[5], compressionRatioResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBreakPoints[6], compressionRatioResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBreakPoints[7], compressionRatioResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of breakpoints')
    plt.ylabel('Compression ratio')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(2)
    plt.plot(numberOfBreakPoints[0], MeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBreakPoints[1], MeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBreakPoints[2], MeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBreakPoints[3], MeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBreakPoints[4], MeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBreakPoints[5], MeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBreakPoints[6], MeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBreakPoints[7], MeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of breakpoints')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(3)
    plt.plot(numberOfBreakPoints[0], baseMeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBreakPoints[1], baseMeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBreakPoints[2], baseMeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBreakPoints[3], baseMeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBreakPoints[4], baseMeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBreakPoints[5], baseMeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBreakPoints[6], baseMeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBreakPoints[7], baseMeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of breakpoints')
    plt.ylabel('Mean absolute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(4)
    plt.plot(numberOfBreakPoints[0], compressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBreakPoints[1], compressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBreakPoints[2], compressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBreakPoints[3], compressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBreakPoints[4], compressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBreakPoints[5], compressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBreakPoints[6], compressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBreakPoints[7], compressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of breakpoints')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.figure(5)
    plt.plot(numberOfBreakPoints[0], decompressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBreakPoints[1], decompressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBreakPoints[2], decompressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBreakPoints[3], decompressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBreakPoints[4], decompressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBreakPoints[5], decompressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBreakPoints[6], decompressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBreakPoints[7], decompressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of breakpoints')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.show() 

if setting==2:
    maxNumsIn=[[]*1 for i in range(15)]

    breakpoints=4
    for truncateBits in range(0, 29, 4):
        for maxNums in range(100, 1501, 100):
            if maxNums == 1500: maxNums=1423
            numsIn=readNums("synthetic_1.csv", 1, maxNums)
            
            results = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)
            results2 = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)  # I run the function 3 times and record the mean to increase reliability of the data obtained
            results3 = compressAndFindRatio(numsIn, breakpoints+1, truncateBits, fast)

            maxNumsIn[resultCtr].append(maxNums)
            compressionRatioResults[resultCtr].append((results[0]+results2[0]+results3[0])/3)
            MeanAbsoluteErrorResults[resultCtr].append((results[1]+results2[1]+results3[1])/3)
            baseMeanAbsoluteErrorResults[resultCtr].append((results[2]+results2[2]+results3[2])/3)
            compressTimeResults[resultCtr].append((results[3]+results2[3]+results3[3])/3)
            decompressTimeResults[resultCtr].append((results[4]+results2[4]+results3[4])/3)

            #print("TRUNCATED BITS: "+str(truncateBits)+". BREAKPOINTS: "+str(breakpoints)+". COMPRESSION RATIO: "+str(compressionRatioResults[resultCtr][-1])+". DECOMPRESSED MAE: " +str(MeanAbsoluteErrorResults[resultCtr][-1])+". BASE MAE: "+str(baseMeanAbsoluteErrorResults[resultCtr][-1])+". EXECUTION TIME: "+str(timeResults[resultCtr][-1]))

        resultCtr+=1


    # GRAPH OBTAINED RESULTS
    plt.figure(1)
    plt.plot(maxNumsIn[0], compressionRatioResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], compressionRatioResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], compressionRatioResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], compressionRatioResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], compressionRatioResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(maxNumsIn[5], compressionRatioResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(maxNumsIn[6], compressionRatioResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(maxNumsIn[7], compressionRatioResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Compression ratio')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(2)
    plt.plot(maxNumsIn[0], MeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], MeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], MeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], MeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], MeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(maxNumsIn[5], MeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(maxNumsIn[6], MeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(maxNumsIn[7], MeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Mean absolute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(3)
    plt.plot(maxNumsIn[0], baseMeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], baseMeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], baseMeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], baseMeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], baseMeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(maxNumsIn[5], baseMeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(maxNumsIn[6], baseMeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(maxNumsIn[7], baseMeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Mean absolute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(4)
    plt.plot(maxNumsIn[0], compressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], compressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], compressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], compressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], compressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(maxNumsIn[5], compressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(maxNumsIn[6], compressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(maxNumsIn[7], compressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.figure(5)
    plt.plot(maxNumsIn[0], decompressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], decompressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], decompressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], decompressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], decompressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(maxNumsIn[5], decompressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(maxNumsIn[6], decompressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(maxNumsIn[7], decompressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.show()