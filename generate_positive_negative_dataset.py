#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Wenyi Qin
#
# Created:     03/05/2018
# Copyright:   (c) Wenyi Qin 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# refseq encode 2

import re
filedir = '/home/xw379/project/miR_project/'
seq_list = []
y_list = []
y_file = open('/home/xw379/project/miR_project/mir_datasets/merge_mir.txt','r')

for eachline in y_file:
    eachline = eachline.strip()
    items = re.split('\t', eachline)
    mRNA = items[0]
    mRNA_file = mRNA + '.txt'
    miRNA = items[2]
    y = int(items[3])
    if miRNA == 'mir181' and y != 0:
        #print(mRNA)
        seq_pd = pd.read_table(filedir + 'refseq_seq_encode2/%s.txt' % (mRNA), header=None)
        seq_np = seq_pd.as_matrix()
        r = np.zeros((4, seq_np.shape[1]))
        t = np.concatenate((seq_np, r), axis = 0)
        if t.shape[1] == 1000:
            seq_list.append(t)
            y_list.append(y)
y_file.close()

s = pd.Series(y_list)
one_hot_y = pd.get_dummies(s)
seqlist_train = seq_list[0:800]
seqlist_test = seq_list[800:]
y_train = one_hot_y[0:800].as_matrix()
y_test = one_hot_y[800:].as_matrix()
X_train = np.array(seqlist_train)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1],X_train.shape[2]))
Y_train = y_train
model.fit(X_train, Y_train, verbose = 1, epochs=35)