import csv
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
import pwlf

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


def compressAndFindRatio(baseBits):
    numsIn, max, min = readNums("randomNumbers.csv")
    compressedLength = 32
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
        compressedNums.append(compressedNum)


    #print("Original numsIn:", numsIn[20100:20120])
    #print("Compressed nums:", compressedNums[20100:20120])
    #print("Bases dictionary:", basesDictionary)

    # Calculate compression ratio
    original_size = sum(len(convertFloatToBin(num)) for num in numsIn)
    compressed_size = ((sum(len(num) for num in compressedNums)) + (len(basesDictionary) * (numBaseBitsRequired + len(list(basesDictionary.values())[0]))))
    compression_ratio = original_size / compressed_size

    print("NUMBER OF BASE BITS: " + str(baseBits) +". COMPRESSION RATIO ACHIEVED: " + str(compression_ratio))

    # Decompression
    decompressedNums = [decompress(num, basesDictionary, numBaseBitsRequired) for num in compressedNums]
    decompressedFloats = [convertBinToFloat(num) for num in decompressedNums]

    with open('decompressedVals.csv', 'w') as file:
        csvwriter = csv.writer(file, delimiter='\n')
        csvwriter.writerow(decompressedFloats)

    #print("Decompressed bits: ", decompressedNums)
    #print("Decompressed nums:", decompressedFloats)
    return compression_ratio


bitsToCompressionRatioDict = {}
for n in range(0,33):
   bitsToCompressionRatioDict[n] = compressAndFindRatio(n)

xVals = bitsToCompressionRatioDict.keys()
yVals = bitsToCompressionRatioDict.values()

plt.plot(xVals, yVals)
plt.title('Number of base bits Vs compression ratio achieved')
plt.xlabel('Number of base bits')
plt.ylabel('Compression ratio')
plt.show()

xVals = []
yVals = []