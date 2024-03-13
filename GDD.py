import csv
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf
import time

# Read in data from csv file
def readNums(filename, rowIndexOfValueToRead):
    file = csv.reader(open(filename, 'r'))
    numsIn = []
    for row in file:
        try:
            num = float(row[rowIndexOfValueToRead])
            numsIn.append(num)
        except ValueError:
            print('error')
            continue
    return numsIn

# Provide a 'uniqueness' score of the provided dataset by finding ratio of unique values to total values
def uniquenessScore(data):
    uniqueValues = len(np.unique(data))
    totalCount = len(data)
    if uniqueValues == 1:
        return 0.0 # If only one unique value in dataset, return score of 0 (as this is the minimum 'uniqueness' possible)
    if uniqueValues == totalCount:
        return 1.0 # If every value is unique, return a score of 1 (as data set is completely unique)
    uniqueness = uniqueValues / totalCount
    return totalCount, uniqueValues, uniqueness

    
# Convert floating point number into binary representation
def convertFloatToBin(floatIn):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floatIn))

# Convert binary to floating point number
def convertBinToFloat(binIn):
    return struct.unpack('!f',struct.pack('!I', int(binIn, 2)))[0]

# Decompress compressed values
def decompress(compressedNum, basesDictionary, numBaseBitsRequired):
    base_bits = compressedNum[:numBaseBitsRequired]
    base = basesDictionary[base_bits]

    decompressedNum = str(compressedNum).replace(base_bits, base, 1) # Replace the dictionary reference bits in the compressed value with the value stored in the dictionary
    return decompressedNum

# Compress input data and find compression ratio
def compressAndFindRatio(baseBits, compressedLength, numsIn):
    start = time.time() # Timer to find time taken to execute
    bases = set() # Used to find number of unique bases as to calculate minimum number of bits required to represent all bases
    basesDictionary = {} # Used to map dictionary references to base values
    ctr = 0 # Tracks number of bases found (for use in calculating number of bits required to represent all bases)
    compressedNums = [] # Stores the compressed version of each input number
    numsInAsBits = [] # Stores the input floats as their 32-bit binary representation (to prevent repeated conversion)
    length = len(numsIn) # Length of input data

    for num in numsIn:
        bits = convertFloatToBin(num)
        numsInAsBits.append(bits)
        base = bits[:baseBits]
                                # This loop finds the number of unique bases so that the number of bits required to represent all bases can be calculated
        if base not in bases:
            bases.add(base)
            ctr += 1

    # Find number of bits required to store all bases (round up to nearest whole number)
    if(ctr < 2):
        numBaseBitsRequired = 1
    else:
        numBaseBitsRequired = math.ceil(math.log2(ctr))  # Take log2 of number of unique bases to find minimum number of bits required to represent in binary

    ctr = 0 # Reset base counter

    for i in range(0,length):
        bits = numsInAsBits[i]
        base = bits[:baseBits]  # Retrieve current input number in binary & take base bits

        if base not in basesDictionary.values(): # Map bases dictionary references to values
            binaryCtr = format(ctr, '0'+str(numBaseBitsRequired)+'b')  # Ensure binary representation has fixed length
            basesDictionary[binaryCtr] = base
            ctr+=1

        # Replace base with the correct dictionary reference
        compressedNum = bits.replace(base, [key for key, value in basesDictionary.items() if value == base][0], 1)
        compressedNum = compressedNum[:compressedLength]
        compressedNums.append(compressedNum)

    # Calculate compression ratio
    original_size = sum(len(num) for num in numsInAsBits) # Original size (of only data, no metadata/overhead) is just the sum of the number of bits representing each number
    compressed_size = ((sum(len(num) for num in compressedNums)) + (len(basesDictionary) * (numBaseBitsRequired + len(list(basesDictionary.values())[0]))))
    # Compressed size is the sum of the length of all compressed values + the size of the base dictionary 
    compressionRatio = original_size / compressed_size # Finds compression ratio: ratio > 1 = compression achieved

    # Decompression
    decompressedNums = [decompress(num, basesDictionary, numBaseBitsRequired) for num in compressedNums] # Decompresses each compressed value: should produce values identical/close to input values
    for i in range(0, len(decompressedNums)): decompressedNums[i] = decompressedNums[i] + '0'*(32-len(decompressedNums[i]))
    # ^ Append each decompressed value with 0's such that they match the expected 32-bit representation (for conversion into float)
    decompressedFloats = [convertBinToFloat(num) for num in decompressedNums] # Convert each decompressed value from binary into float - decompression complete

    # with open('decompressedVals.csv', 'w') as file:
    #     csvwriter = csv.writer(file, delimiter='\n')          # Used in testing to compare input and decompressed data
    #     csvwriter.writerow(decompressedFloats)

    #print("Decompressed bits: ", decompressedNums)
    #print("Decompressed nums:", decompressedFloats)
        
    #  Find mean average error from decompressed values against input values
    totalError=0
    for i in range(0, length):
        observedVal = numsIn[i]
        compressedVal = decompressedFloats[i]
        totalError += abs(observedVal-compressedVal)

    meanError = totalError/length


    # Find mean average error from base values against input values
    totalBaseError=0
    for i in range(0, length):
        observedVal = numsIn[i]
        baseVal = basesDictionary[(compressedNums[i])[:numBaseBitsRequired]]
        baseVal = convertBinToFloat(baseVal+('0'*(32-baseBits)))
        totalBaseError += abs(observedVal-baseVal)

    totalBaseError = totalBaseError/length

    end = time.time()
    timeTaken = end-start # end time - start time = time taken for function execution in seconds
    #print("NUMBER OF BASE BITS: " + str(baseBits) +". COMPRESSION RATIO ACHIEVED: " + str(compressionRatio) + ". MEAN SQUARED ERROR: " + str(meanError))
    return compressionRatio, meanError, totalBaseError, timeTaken


# Used to hold the results from algorithm execution (for graphing)
compressionRatioResults = [[]*1 for i in range(8)]
meanAverageErrorResults=[[]*1 for i in range(8)]
baseMeanAverageErrorResults=[[]*1 for i in range(8)]
timeResults=[[]*1 for i in range(8)]
numberOfBaseBits=[[]*1 for i in range(8)]

#baseBits = int(input("Please enter number of bits from the input data to be assigned to the base (max 32): "))
#compressedLength = int(input("Please enter maximum number of bits used to represent the compressed values (max 32): "))
numsIn = readNums("randomNumbers.csv", 0) # Read in input data once & provide to each compression function execution (prevent repeated "readNums" execution)
totalValues, uniqueValues, similarity = uniquenessScore(numsIn) # Provide the 'uniqueness' score for the input data set

resultCtr=0
compressedLength=32

# Obtain results
for compressedLength in range(32, 0, -4): # Decrement maximum compressed length by 4 bits each cycle
    for bitsInBase in range(1, compressedLength+1): # For each compressedLength value, collect data for each number of base bits 1-compressedLength (e.g 1-32 bits inclusive)
        results = compressAndFindRatio(bitsInBase, compressedLength, numsIn)
        results2 = compressAndFindRatio(bitsInBase, compressedLength, numsIn)  # I run the function 3 times and record the mean to increase reliability of the data obtained
        results3 = compressAndFindRatio(bitsInBase, compressedLength, numsIn)
        numberOfBaseBits[resultCtr].append(bitsInBase)
        compressionRatioResults[resultCtr].append((results[0]+results2[0]+results3[0])/3)
        meanAverageErrorResults[resultCtr].append((results[1]+results2[1]+results3[1])/3)
        baseMeanAverageErrorResults[resultCtr].append((results[2]+results2[2]+results3[2])/3)
        timeResults[resultCtr].append((results[3]+results2[3]+results3[3])/3)

        #print("MAX LENGTH: "+str(compressedLength)+". BITS IN BASE: "+str(bitsInBase)+". COMPRESSION RATIO: "+str(compressionRatioResults[resultCtr][-1])+". DECOMPRESSED MAE: " +str(meanAverageErrorResults[resultCtr][-1])+". BASE MAE: "+str(baseMeanAverageErrorResults[resultCtr][-1])+". EXECUTION TIME: "+str(timeResults[resultCtr][-1]))

    resultCtr+=1

print(numberOfBaseBits)
# GRAPH OBTAINED RESULTS
plt.figure(1)
plt.plot(numberOfBaseBits[0], compressionRatioResults[0], 'x-', color='b', label='max 32 bits')
plt.plot(numberOfBaseBits[1], compressionRatioResults[1], 'x-', color='g', label='max 28 bits')
plt.plot(numberOfBaseBits[2], compressionRatioResults[2], 'x-', color='r', label='max 24 bits')
plt.plot(numberOfBaseBits[3], compressionRatioResults[3], 'x-', color='c', label='max 20 bits')
plt.plot(numberOfBaseBits[4], compressionRatioResults[4], 'x-', color='m', label='max 16 bits')
plt.plot(numberOfBaseBits[5], compressionRatioResults[5], 'x-', color='y', label='max 12 bits')
plt.plot(numberOfBaseBits[6], compressionRatioResults[6], 'x-', color='k', label='max 8 bits')
plt.plot(numberOfBaseBits[7], compressionRatioResults[7], 'x-', color='#643B9F', label='max 4 bits')
plt.title('Number of Bits in Base Vs Compression Ratio Achieved')
plt.xlabel('Number of Bits in Base')
plt.ylabel('Compression Ratio')
plt.legend(loc="upper right")
plt.grid()

plt.figure(2)
plt.plot(numberOfBaseBits[0], meanAverageErrorResults[0], 'x-', color='b', label='max 32 bits')
plt.plot(numberOfBaseBits[1], meanAverageErrorResults[1], 'x-', color='g', label='max 28 bits')
plt.plot(numberOfBaseBits[2], meanAverageErrorResults[2], 'x-', color='r', label='max 24 bits')
plt.plot(numberOfBaseBits[3], meanAverageErrorResults[3], 'x-', color='c', label='max 20 bits')
plt.plot(numberOfBaseBits[4], meanAverageErrorResults[4], 'x-', color='m', label='max 16 bits')
plt.plot(numberOfBaseBits[5], meanAverageErrorResults[5], 'x-', color='y', label='max 12 bits')
plt.plot(numberOfBaseBits[6], meanAverageErrorResults[6], 'x-', color='k', label='max 8 bits')
plt.plot(numberOfBaseBits[7], meanAverageErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
plt.title('Number of Base Bits Vs Mean Average Error against decompressed values')
plt.xlabel('Number of Base Bits')
plt.ylabel('Mean Average Error')
plt.legend(loc="upper right")
plt.grid()

plt.figure(3)
plt.plot(numberOfBaseBits[0], baseMeanAverageErrorResults[0], 'x-', color='b', label='max 32 bits')
plt.plot(numberOfBaseBits[1], baseMeanAverageErrorResults[1], 'x-', color='g', label='max 28 bits')
plt.plot(numberOfBaseBits[2], baseMeanAverageErrorResults[2], 'x-', color='r', label='max 24 bits')
plt.plot(numberOfBaseBits[3], baseMeanAverageErrorResults[3], 'x-', color='c', label='max 20 bits')
plt.plot(numberOfBaseBits[4], baseMeanAverageErrorResults[4], 'x-', color='m', label='max 16 bits')
plt.plot(numberOfBaseBits[5], baseMeanAverageErrorResults[5], 'x-', color='y', label='max 12 bits')
plt.plot(numberOfBaseBits[6], baseMeanAverageErrorResults[6], 'x-', color='k', label='max 8 bits')
plt.plot(numberOfBaseBits[7], baseMeanAverageErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
plt.title('Number of Base Bits Vs Mean Average Error against base values')
plt.xlabel('Number of Base Bits')
plt.ylabel('Mean Average Error')
plt.legend(loc="upper right")
plt.grid()

plt.figure(4)
plt.plot(numberOfBaseBits[0], timeResults[0], 'x-', color='b', label='max 32 bits')
plt.plot(numberOfBaseBits[1], timeResults[1], 'x-', color='g', label='max 28 bits')
plt.plot(numberOfBaseBits[2], timeResults[2], 'x-', color='r', label='max 24 bits')
plt.plot(numberOfBaseBits[3], timeResults[3], 'x-', color='c', label='max 20 bits')
plt.plot(numberOfBaseBits[4], timeResults[4], 'x-', color='m', label='max 16 bits')
plt.plot(numberOfBaseBits[5], timeResults[5], 'x-', color='y', label='max 12 bits')
plt.plot(numberOfBaseBits[6], timeResults[6], 'x-', color='k', label='max 8 bits')
plt.plot(numberOfBaseBits[7], timeResults[7], 'x-', color='#643B9F', label='max 4 bits')
plt.title('Number of Base Bits Vs Algorithm Execution Time')
plt.xlabel('Number of Base Bits')
plt.ylabel('Execution Time (s)')
plt.legend(loc="lower right")
plt.grid()

print("DATASET LEGTH: " + str(totalValues) + ". UNIQUE VALUES: " + str(uniqueValues) + ". UNIQUENESS RATIO: " + str(similarity))
plt.show() 