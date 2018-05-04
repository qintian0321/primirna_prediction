#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Wenyi Qin
#
# Created:     09/04/2018
# Copyright:   (c) Wenyi Qin 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
#import matplotlib.pyplot as plt
import os.path
import os
import sys
import math
input_file=sys.argv[1]
length=int(sys.argv[2])
max_length=int(sys.argv[3])
output_file_name=sys.argv[4]
index_shift=int(math.floor((max_length-length)/2))
struct_file=open(input_file).readlines()[3:]
count_matrix=np.zeros(shape=(max_length,max_length))#numpy array
for each_line in struct_file:
    count_line=each_line.rstrip().split()
    if count_line[0]!="Structure":
        contact_x=int(count_line[0])+index_shift#index shift to make it same size
        contact_y=int(count_line[1])+index_shift
        k=int(count_line[2])#how many contacts
        for each in range(k):
            count_matrix[contact_x-1+each,contact_y-1-each]+=1#increase the count by 1
            count_matrix[contact_y-1-each,contact_x-1+each]+=1#symmetric
count_matrix_new=count_matrix/1000#convert it to probablity
np.savetxt(output_file_name,count_matrix_new)
##    plt.matshow(count_matrix_new)
##    plt.savefig(output_file_name)
##    plt.close()
#plt.show()
#plt.imshow(count_matrix_new)
#plt.show()
