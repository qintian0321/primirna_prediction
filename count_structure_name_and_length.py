#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Wenyi Qin
#
# Created:     15/04/2018
# Copyright:   (c) Wenyi Qin 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import string
#all_ref_seq=open("D:/Research_data/Jun_lu/GRCh38_refseq_frag_400_tile_200.fa","r").readlines()
mirna_seq=open("C:/Research_data/JunLu_microRNA/hsa_v21_miR_pre_200bp_surround.fa","r").readlines()
output=open("C:/Research_data/JunLu_microRNA/positive_mirna.txt","w")
for i in range(0,len(mirna_seq),2):
    print i
    title=string.split(string.replace(string.strip(mirna_seq[i]),">",""))[0]
    length=len(string.strip(mirna_seq[i+1]))
    print >>output, title, length
output.close()