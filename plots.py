# -*- coding: utf-8 -*-
"""
Script to print the learning rates based on the logs from the spark
programme. It allows both to print the different learning rates results
for a single updater as well as printing the best learning rate
comparing all the updaters
"""
import matplotlib.pyplot as plt
"""
#-------------------------------------------------------------------------------

#COMPARISON OF RESULTS OF 7 LEARNING RATES FOR EVERY UPDATER

#Read file
f = open("svm_mushroom_2.txt","r")
contents = f.read()
f.close()

res=[]
l=[]
choices=[]
for row in contents.split("\n"):
    line = row.split(" ")
    if line[0]=="Loss":
        l.append(float(line[-1]))
    elif (len(l)>0):
        res.append(l[:])
        l=[]
        choices.append(float(line[11]))

for i in range (10):
    #Plot 1 esp
    plt.figure()
    l1,=plt.plot(list(range(len(res[7*i]))),res[7*i], color='g')
    l2,=plt.plot(list(range(len(res[7*i+1]))),res[7*i+1], color='b')
    l3,=plt.plot(list(range(len(res[7*i+2]))),res[7*i+2], color='r')
    l4,=plt.plot(list(range(len(res[7*i+3]))),res[7*i+3], color='y')
    l5,=plt.plot(list(range(len(res[7*i+4]))),res[7*i+4], color='grey')
    l6,=plt.plot(list(range(len(res[7*i+5]))),res[7*i+5], color='orchid')
    l7,=plt.plot(list(range(len(res[7*i+6]))),res[7*i+6], color='k')
    plt.ylim([0,0.2])
    plt.xlim([0,125])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    st='Logistic regression - svmguide1 dataset for '+leg[i]+' with different learning rates'
    plt.title(st)
    plt.legend([l1, l2,l3, l4, l5, l6, l7], [choices[7*i], choices[7*i+1], choices[7*i+2],choices[7*i+3],choices[7*i+4],choices[7*i+5],choices[7*i+6]],bbox_to_anchor=(1.04,1), loc="upper left")


#-------------------------------------------------------------------------------
"""
#SELECT BETWEEN LEARNING RATES AND PLOT THE BEST ONES FOR EACH UPDATER

#Read file
f = open("svm_mushroom_2_decay.txt","r")
contents = f.read()
f.close()

#HOW MANY LEARNING RATES WERE TRIED PER UPDATER
nhyper=7 
#IN WHICH ITERATION TO COMPARE ACCURACIES TO DECIDE BETWEEN RATES
itcrit=20

res=[]
l=[]
criteria=1 
#Contains the learning rates chosen for information purposes
choices=[] 
c=0
for row in contents.split("\n"):
    line = row.split(" ")
    if (line[0]=="Loss"):
        l.append(float(line[-1]))
    elif (len(l)>0):
        if(c%nhyper==0):
            if(len(l)>=itcrit): criteria=l[itcrit-1]
            else: criteria=1
            res.append(l[:])
            choices.append(float(line[11]))
        elif(len(l)>=itcrit and l[itcrit-1]<criteria):
            criteria=l[itcrit-1]
            res.pop()
            res.append(l[:])
            choices.pop()
            choices.append(float(line[11]))
        c=c+1
        l=[]


plt.figure()
l1,=plt.plot(list(range(len(res[0]))),res[0], color='g')
l2,=plt.plot(list(range(len(res[1]))),res[1], color='b')
l3,=plt.plot(list(range(len(res[2]))),res[2], color='r')
l4,=plt.plot(list(range(len(res[3]))),res[3], color='y')
l5,=plt.plot(list(range(len(res[4]))),res[4], color='grey')
l6,=plt.plot(list(range(len(res[5]))),res[5], color='orchid')
plt.ylim([0,0.2])
plt.xlim([0,125])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('SVM - mushroom dataset with decay')
plt.legend([l1, l2,l3,l4,l5, l6], ['Simple', 'Momentum', 'Nesterov','Adagrad','Adadelta','RMSprop'],bbox_to_anchor=(1.04,1), loc="upper left")

plt.figure()
l1,=plt.plot(list(range(len(res[0]))),res[0], color='g')
l7,=plt.plot(list(range(len(res[6]))),res[6], color='k')
l8,=plt.plot(list(range(len(res[7]))),res[7], color='c')
l9,=plt.plot(list(range(len(res[8]))),res[8], color='purple')
l10,=plt.plot(list(range(len(res[9]))),res[9], color='orange')
plt.ylim([0,0.2])
plt.xlim([0,125])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('SVM - mushroom dataset with decay')
plt.legend([l1,l7,l8,l9,l10], ['Simple','Adam','Adamax','Nadam','AMSgrad'],bbox_to_anchor=(1.04,1), loc="upper left")
