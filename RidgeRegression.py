import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
Inf = 9999999999999999999

np.set_printoptions(threshold=np.nan)
#This function plot each ROC curve with different lambda on each iteration
def plotROC(TPR,FPR, iter, lam):
    plt.ion()
    titleName = 'Iter_' + str(iter) + '_lambda_' + str(lam)
    NamefiltTpye = 'output\\' + titleName + '.png'
    if not os.path.exists('output'):
        os.makedirs('output')
    plt.figure(1)
    plt.plot([0, 1.05], [0, 1.05], 'k--')
    plt.plot(FPR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titleName)
    plt.savefig(NamefiltTpye)
    plt.show()
    plt.close() # comment this line if you want to see each image in program or each image will be store at output folder


# start read training file and test file and then compute Ridge Regression
def start():
    inputData = pd.read_csv('MNIST_15_15.csv',header=None) # read MNiST data
    inputDataList = inputData.values
    label = pd.read_csv('MNIST_LABEL.csv', header=None) # read MNIST label
    labelList = label.values
    lam = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.00000000001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, Inf] #lambda
    count = 0
    fp = 0
    TPR = []
    FPR = []

    for i in range(0,inputDataList.shape[0]): # normalization
        for j in range(0, inputDataList.shape[1]):
            if inputDataList[i][j] != 0:
                inputDataList[i][j] = 1
            else:
                inputDataList[i][j] = 0

    K = 10
    for k in range(K): # k fold validation
        crossTrainingSet = [x for i, x in enumerate(inputDataList) if i % K != k] # training dataset
        crossTestSet = [x for i, x in enumerate(inputDataList) if i % K == k] # training label
        crossTrainingLabel = [x for i, x in enumerate(labelList) if i % K != k] # test dataset
        crossTestLabel = [x for i, x in enumerate(labelList) if i % K == k] # test label

        print("iter: %s" % str(k + 1))


        X = np.asmatrix(crossTrainingSet)
        X = np.insert(X, 0, np.ones(len(crossTrainingSet)), 1)
        Y = np.asmatrix(crossTrainingLabel)

        for i in range(0, len(lam)):


            R = np.eye(np.shape(X)[1]) * lam[i] # ridge
            B = (X.T * X + R).I * X.T * Y # optimization
            actual = []
            pred = []
            for j in range(0, len(crossTestSet)):
                predResult = calculateRR(B, crossTestSet[j]) # calculate Ridge Regression
                x = predResult
                y = crossTestLabel[j][0]
                if y > 5:
                    y = 1
                else:
                    y = 0
                actual.append(y)
                pred.append(x)

                x = int(round(predResult))
                y = crossTestLabel[j][0]
                if x > 5:
                    x = 1
                else:
                    x = 0

                if y > 5:
                    y = 1
                else:
                    y = 0

                if x == 1 and y == 1:
                    count = count + 1
                elif x == 0 and y == 1:
                    fp = fp + 1
            tpr = count / len(crossTestLabel)
            TPR.append(tpr)
            fpr = fp / len(crossTestLabel)
            FPR.append(fpr)
            if lam[i] > 100:
                print('Lamda: Inf')
            else:
                print('Lamda: %.2f' % lam[i])
            print('TPR: %.4f' % tpr)
            print('FPR: %.4f' % fpr)
            #plot ROC curve and save image at output folder
            # if lam[i] > 100:
            #     calculateTPRFPR(actual, pred, k + 1, 'Inf')
            # else:
            #     calculateTPRFPR(actual, pred, k + 1, lam[i])
            count = 0
            fp = 0






# This function compute the Ridge Regression
def calculateRR(B, pixelValue):
    intercept = B[0, 0]
    RR = 0.0
    for i in range(0, len(pixelValue)):
        RR = RR + B[i+1, 0] *pixelValue[i]
    RR = RR + intercept
    return RR
	
# this function compute TPR,FTR and then plot the ROC curve
def calculateTPRFPR(label, prediction, iter, lam):
    FPR = [] # FPR list
    TPR = [] # TPR list
    neg = float(len(label) - sum(label))
    total = float(sum(label))
    for x in np.arange(min(prediction), max(prediction), 1.0/len(prediction)):
        tpr = 0.0
        fpr = 0.0
        for y in range(len(prediction)):
            if (prediction[y] > x) and (label[y] == 1):
                tpr = tpr + 1 #true positive
            if (prediction[y] > x) and (label[y] == 0):
                fpr = fpr + 1 #false positive
        TPR.append(tpr / total)
        FPR.append(fpr / neg)
    plotROC(TPR,FPR, iter, lam)

if __name__ == "__main__":
    start()