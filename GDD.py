import csv
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf
import time

#read in data from csv file
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
    
#convert floating point number into binary representation
def convertFloatToBin(floatIn):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', floatIn))

#convert binary to floating point number
def convertBinToFloat(binIn):
    return struct.unpack('!f',struct.pack('!I', int(binIn, 2)))[0]

#decompress compressed values
def decompress(compressedNum, basesDictionary, numBaseBitsRequired):
    base_bits = compressedNum[:numBaseBitsRequired]
    base = basesDictionary[base_bits]

    decompressedNum = str(compressedNum).replace(base_bits, base, 1)
    return decompressedNum


def compressAndFindRatio(baseBits, compressedLength):
    start = time.time()
    numsIn, max, min = readNums("randomNumbers3.csv")
    bases = set()
    basesDictionary = {}
    ctr = 0
    compressedNums = []

    for num in numsIn:
        bits = convertFloatToBin(num)
        base = bits[:baseBits]

        if base not in bases:
            bases.add(base)
            ctr += 1

    # Find number of bits required to store all bases (round up to nearest whole number)
    if(ctr < 2):
        numBaseBitsRequired = 1
    else:
        numBaseBitsRequired = math.ceil(math.log2(ctr))

    newCtr = -1

    for num in numsIn:
        bits = convertFloatToBin(num)
        base = bits[:baseBits]

        if base not in basesDictionary.values():
            newCtr+=1
            binaryCtr = format(newCtr, '0'+str(numBaseBitsRequired)+'b')  # Ensure binary representation has fixed length
            basesDictionary[binaryCtr] = base

        # Replace base with the correct dictionary reference
        compressedNum = bits.replace(base, [key for key, value in basesDictionary.items() if value == base][0], 1)
        compressedNum = compressedNum[:compressedLength]
        compressedNums.append(compressedNum)

    print("COMPRESSED LENGTH:", compressedLength, ". # BITS IN BASE: ", baseBits, ". # BASE BITS REQUIRED: ", numBaseBitsRequired)
    print("Final numsIn: " + str(numsIn[-1]) + "  UNCOMPRESSED BITS: " + str(convertFloatToBin(numsIn[-1])))
    print("Compressed: " + str(compressedNums[-1] + "  LENGTH: " + str(len(compressedNums[-1]))))
    print("BASE:", basesDictionary[compressedNums[-1][:numBaseBitsRequired]], "VALUE:", convertBinToFloat((basesDictionary[compressedNums[-1][:numBaseBitsRequired]]) + '0'*(32-baseBits)))
    print()

    # Calculate compression ratio
    original_size = sum(len(convertFloatToBin(num)) for num in numsIn)
    compressed_size = ((sum(len(num) for num in compressedNums)) + (len(basesDictionary) * (numBaseBitsRequired + len(list(basesDictionary.values())[0]))))
    compressionRatio = original_size / compressed_size

    # Decompression
    # print("FIRST BASE: " + str(basesDictionary['0'*numBaseBitsRequired]+'0'*(32-baseBits)))
    decompressedNums = [decompress(num, basesDictionary, numBaseBitsRequired) for num in compressedNums]
    for i in range(0, len(decompressedNums)): decompressedNums[i] = decompressedNums[i] + '0'*(32-len(decompressedNums[i]))
    #print(32-(compressedLength+(baseBits-1)))
    #print(decompressedNums[0] + " len: " + str(len(decompressedNums[5]))) 
    decompressedFloats = [convertBinToFloat(num) for num in decompressedNums]

    with open('decompressedVals.csv', 'w') as file:
        csvwriter = csv.writer(file, delimiter='\n')
        csvwriter.writerow(decompressedFloats)

    #print("Decompressed bits: ", decompressedNums)
    #print("Decompressed nums:", decompressedFloats)
        
    #  Find mean squared error
    n = len(numsIn)
    #print(str(convertFloatToBin(numsIn[0])) +" "+ str(convertFloatToBin(numsIn[1])) +" "+ str(convertFloatToBin(numsIn[2])))
    totalError=0
    for i in range(0, n):
        observedVal = numsIn[i]
        compressedVal = decompressedFloats[i]
        totalError += abs(observedVal-compressedVal)

    meanError = totalError/n


    # Find mean squared error FROM ONLY BASES
    n = len(numsIn)
    totalBaseError=0
    for i in range(0, n):
        observedVal = numsIn[i]
        baseVal = basesDictionary[(compressedNums[i])[:numBaseBitsRequired]]
        baseVal = convertBinToFloat(baseVal+('0'*(32-baseBits)))
        totalBaseError += abs(observedVal-baseVal)

    totalBaseError = totalBaseError/n

    end = time.time()
    timeTaken = end-start
    #print("NUMBER OF BASE BITS: " + str(baseBits) +". COMPRESSION RATIO ACHIEVED: " + str(compressionRatio) + ". MEAN SQUARED ERROR: " + str(meanError))
    return compressionRatio, meanError, totalBaseError, timeTaken


compressionRatioResults = []
meanAverageErrorResults=[]
baseMeanAverageErrorResults=[]
timeResults=[]
numberOfBaseBits=[]
#baseBits = int(input("Please enter number of bits from the input data to be assigned to the base (max 32): "))
compressedLength = int(input("Please enter maximum number of bits used to represent the compressed values (max 32): "))

for bitsInBase in range(1, 26):
   results = compressAndFindRatio(bitsInBase, compressedLength)
   numberOfBaseBits.append(bitsInBase)
   compressionRatioResults.append(results[0])
   meanAverageErrorResults.append(results[1])
   baseMeanAverageErrorResults.append(results[2])
   timeResults.append(results[3])


plt.figure(1)
plt.plot(numberOfBaseBits, compressionRatioResults, '-', label='Compression Ratio')
plt.title('Number of Bits in Base Vs Compression Ratio Achieved')
plt.xlabel('Number of Bits in Base')
plt.ylabel('Compression Ratio')
plt.legend(loc="upper left")

plt.figure(2)
plt.plot(numberOfBaseBits, meanAverageErrorResults, '-', label='Decompressed value MAE')
plt.plot(numberOfBaseBits, baseMeanAverageErrorResults, '-', label='Base value MAE')
plt.title('Number of Base Bits Vs Mean Average Errors recorded')
plt.xlabel('Number of Base Bits')
plt.ylabel('Mean Average Error')
plt.legend(loc="upper left")

plt.figure(3)
plt.plot(numberOfBaseBits, timeResults, '-', label='Execution Time (s)')
plt.title('Number of Base Bits Vs Algorithm Execution Time')
plt.xlabel('Number of Base Bits')
plt.ylabel('Execution Time (s)')
plt.legend(loc="upper left")
plt.show()