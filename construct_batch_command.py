#-------------------------------------------------------------------------------
# Name:        construct_batch_command
# Purpose:
#
# Author:      chris
#
# Created:     24/04/2018
# Copyright:   (c) chris 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import string
def main():
   positive_dir_path="/home/cp832/scratch60/qin/Jun_Lu/hsa_v21_mir_pre_200/"
   negative_dir_path="/home/cp832/scratch60/qin/Jun_Lu/SFold_1000_struct/"
   output=open("C:/Research_data/JunLu_microRNA/generate_miRNA_contact_batch.txt","w")
   positive_file=open("C:/Research_data/JunLu_microRNA/positive_miRNA.txt").readlines()
   negative_file=open("C:/Research_data/JunLu_microRNA/negative_miRNA.txt").readlines()
   output_positive_path="/home/cp832/scratch60/qin/Jun_Lu/positive_contact_matrix"
   output_negative_path="/home/cp832/scratch60/qin/Jun_Lu/positive_contact_matrix"
   max_length=580
   for each_line in positive_file:
     pos_info=string.split(string.strip(each_line))
     command="python convert_structure_to_contact_matrix.py"+" "+positive_dir_path+pos_info[0]+\
     "/sample_1000.out "+pos_info[1]+" "+str(max_length)+" "+\
     output_positive_path+"/"+pos_info[0]+".mat.gz"
     print >>output,command
   for each_line in negative_file:
     neg_info=string.strip(each_line)
     command="python convert_structure_to_contact_matrix.py"+" "+negative_dir_path+neg_info+\
     "/sample_1000.out "+"400"+" "+str(max_length)+" "+\
     output_negative_path+"/"+neg_info+".mat.gz"
     print >>output,command
if __name__ == '__main__':
    main()
