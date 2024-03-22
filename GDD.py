import csv
import struct
import math
import matplotlib.pyplot as plt
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
                #print('error')
                continue
            rowNum+=1
        else:
            return numsIn
    return numsIn

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
def compressAndFindRatio(baseBits, compressedLength, numsIn, verbose):
    compressStart = time.time() # Timer to find time taken to execute
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
    
    compressEnd=time.time()
    compressTime = compressEnd-compressStart

    # Calculate compression ratio
    original_size = sum(len(num) for num in numsInAsBits) # Original size (of only data, no metadata/overhead) is just the sum of the number of bits representing each number
    compressed_size = ((sum(len(num) for num in compressedNums)) + 
    (len(basesDictionary) * (numBaseBitsRequired + len(list(basesDictionary.values())[0]))))
    # Compressed size is the sum of the length of all compressed values + the size of the base dictionary 
    compressionRatio = original_size / compressed_size # Finds compression ratio: ratio > 1 = compression achieved

    # Decompression
    decompressStart = time.time()
    decompressedNums = [decompress(num, basesDictionary, numBaseBitsRequired) for num in compressedNums] # Decompresses each compressed value: should produce values identical/close to input values
    for i in range(0, len(decompressedNums)): decompressedNums[i] = decompressedNums[i] + '0'*(32-len(decompressedNums[i]))
    # ^ Append each decompressed value with 0's such that they match the expected 32-bit representation (for conversion into float)
    decompressedFloats = [convertBinToFloat(num) for num in decompressedNums] # Convert each decompressed value from binary into float - decompression complete

    decompressEnd = time.time()
    decompressTime = decompressEnd-decompressStart

    # with open('decompressedVals.csv', 'w') as file:
    #     csvwriter = csv.writer(file, delimiter='\n')          # Used in testing to compare input and decompressed data
    #     csvwriter.writerow(decompressedFloats)

    #print("Decompressed bits: ", decompressedNums)
    #print("Decompressed nums:", decompressedFloats)
        
    #  Find mean abslute error from decompressed values against input values
    totalError=0
    for i in range(0, length):
        observedVal = numsIn[i]
        compressedVal = decompressedFloats[i]
        totalError += abs(observedVal-compressedVal)

    meanError = totalError/length


    # Find mean abslute error from base values against input values
    totalBaseError=0
    for i in range(0, length):
        observedVal = numsIn[i]
        baseVal = basesDictionary[(compressedNums[i])[:numBaseBitsRequired]]
        baseVal = convertBinToFloat(baseVal+('0'*(32-baseBits)))
        totalBaseError += abs(observedVal-baseVal)

    totalBaseError = totalBaseError/length

    #print("NUMBER OF BASE BITS: " + str(baseBits) +". COMPRESSION RATIO ACHIEVED: " + str(compressionRatio) + ". MEAN ABSOLUTE ERROR: " + str(meanError))
    if verbose==1:
        print("DATASET LENGTH: " + str(len(numsIn)) +". DICTIONARY LENGTH: "
            + str(len(basesDictionary)) + ". DICTIONARY SIZE: " + str(len(basesDictionary)
            * (numBaseBitsRequired + len(list(basesDictionary.values())[0])))
            + ". DICTIONARY REFERENCE BITS REQUIRED: " + str(numBaseBitsRequired))

    return compressionRatio, meanError, totalBaseError, compressTime, decompressTime


# Used to hold the results from algorithm execution (for graphing)
compressionRatioResults = [[]*1 for i in range(8)]
MeanAbsoluteErrorResults=[[]*1 for i in range(8)]
baseMeanAbsoluteErrorResults=[[]*1 for i in range(8)]
compressTimeResults=[[]*1 for i in range(8)]
decompressTimeResults = [[]*1 for i in range(8)]
numberOfBaseBits=[[]*1 for i in range(8)]

#baseBits = int(input("Please enter number of bits from the input data to be assigned to the base (max 32): "))
#compressedLength = int(input("Please enter maximum number of bits used to represent the compressed values (max 32): "))
numsIn = readNums("synthetic_1.csv", 1, 1425) # Read in input data once & provide to each compression function execution (prevent repeated "readNums" execution)

resultCtr=0

verbose=0
verboseOuter=0

setting=2

if setting==1:
    # Obtain results
    for compressedLength in range(32, 0, -4): # Decrement maximum compressed length by 4 bits each cycle
        for bitsInBase in range(1, compressedLength+1): # For each compressedLength value, collect data for each number of base bits 1-compressedLength (e.g 1-32 bits inclusive)
            results = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)
            results2 = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)  # I run the function 3 times and record the mean to increase reliability of the data obtained
            results3 = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)
            numberOfBaseBits[resultCtr].append(bitsInBase)
            compressionRatioResults[resultCtr].append((results[0]+results2[0]+results3[0])/3)
            MeanAbsoluteErrorResults[resultCtr].append((results[1]+results2[1]+results3[1])/3)
            baseMeanAbsoluteErrorResults[resultCtr].append((results[2]+results2[2]+results3[2])/3)
            compressTimeResults[resultCtr].append((results[3]+results2[3]+results3[3])/3)
            decompressTimeResults[resultCtr].append((results[4]+results2[4]+results3[4])/3)

            #print("MAX LENGTH: "+str(compressedLength)+". BITS IN BASE: "+str(bitsInBase)+". COMPRESSION RATIO: "+str(compressionRatioResults[resultCtr][-1])+". DECOMPRESSED MAE: " +str(MeanAbsoluteErrorResults[resultCtr][-1])+". BASE MAE: "+str(baseMeanAbsoluteErrorResults[resultCtr][-1])+". EXECUTION TIME: "+str(compressTimeResults[resultCtr][-1]))

        resultCtr+=1

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
    plt.xlabel('Number of bits in base')
    plt.ylabel('Compression ratio')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(2)
    plt.plot(numberOfBaseBits[0], MeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBaseBits[1], MeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBaseBits[2], MeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBaseBits[3], MeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBaseBits[4], MeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBaseBits[5], MeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBaseBits[6], MeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBaseBits[7], MeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of bits in base')
    plt.ylabel('Mean absolute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(3)
    plt.plot(numberOfBaseBits[0], baseMeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBaseBits[1], baseMeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBaseBits[2], baseMeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBaseBits[3], baseMeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBaseBits[4], baseMeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBaseBits[5], baseMeanAbsoluteErrorResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBaseBits[6], baseMeanAbsoluteErrorResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBaseBits[7], baseMeanAbsoluteErrorResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of bits in base')
    plt.ylabel('Mean absolute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(4)
    plt.plot(numberOfBaseBits[0], compressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBaseBits[1], compressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBaseBits[2], compressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBaseBits[3], compressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBaseBits[4], compressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBaseBits[5], compressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBaseBits[6], compressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBaseBits[7], compressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of bits in base')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="upper left")
    plt.grid()

    plt.figure(5)
    plt.plot(numberOfBaseBits[0], decompressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(numberOfBaseBits[1], decompressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(numberOfBaseBits[2], decompressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(numberOfBaseBits[3], decompressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(numberOfBaseBits[4], decompressTimeResults[4], 'x-', color='m', label='max 16 bits')
    plt.plot(numberOfBaseBits[5], decompressTimeResults[5], 'x-', color='y', label='max 12 bits')
    plt.plot(numberOfBaseBits[6], decompressTimeResults[6], 'x-', color='k', label='max 8 bits')
    plt.plot(numberOfBaseBits[7], decompressTimeResults[7], 'x-', color='#643B9F', label='max 4 bits')
    plt.xlabel('Number of bits in base')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="upper right")
    plt.grid()

    plt.show() 

if setting==2:

    maxNumsIn=[[]*1 for i in range(15)]

    bitsInBase=16
    for compressedLength in range(32, bitsInBase-1, -4):
        for maxNums in range(100, 1501, 100):
            if maxNums == 1500: maxNums=1423
            if verboseOuter==1: verbose=1
            numsIn=readNums("synthetic_1.csv", 1, maxNums)
            
            results = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)
            verbose=0
            results2 = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)  # I run the function 3 times and record the mean to increase reliability of the data obtained
            results3 = compressAndFindRatio(bitsInBase, compressedLength, numsIn, verbose)

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
    plt.plot(maxNumsIn[0], compressionRatioResults[0], 'D-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], compressionRatioResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], compressionRatioResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], compressionRatioResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], compressionRatioResults[4], 'x-', color='m', label='max 16 bits')
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
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Mean abslute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(3)
    plt.plot(maxNumsIn[0], baseMeanAbsoluteErrorResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], baseMeanAbsoluteErrorResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], baseMeanAbsoluteErrorResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], baseMeanAbsoluteErrorResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], baseMeanAbsoluteErrorResults[4], 'x-', color='m', label='max 16 bits')
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Mean abslute error')
    plt.legend(loc="upper right")
    plt.grid()

    plt.figure(4)
    plt.plot(maxNumsIn[0], compressTimeResults[0], 'x-', color='b', label='max 32 bits')
    plt.plot(maxNumsIn[1], compressTimeResults[1], 'x-', color='g', label='max 28 bits')
    plt.plot(maxNumsIn[2], compressTimeResults[2], 'x-', color='r', label='max 24 bits')
    plt.plot(maxNumsIn[3], compressTimeResults[3], 'x-', color='c', label='max 20 bits')
    plt.plot(maxNumsIn[4], compressTimeResults[4], 'x-', color='m', label='max 16 bits')
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
    plt.xlabel('Data set size (number of individual entries)')
    plt.ylabel('Execution time (s)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.show()