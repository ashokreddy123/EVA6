### Problem Statement:

​	Improve the given network to -

​		i) achieve 99.4% within 14 epochs (consistent within last 4 epochs)

​		ii) use around 8k parameters



#### **Group - 25** -

​		1) Ashok

​		2)Gokul



**Code 1:**

**Target :**

Setting up the colab file

Setting basic working code 

Setting training and testing loop 

**Results :**

Parameters - 6.38M 

Best Train accuracy - 99.77 (within 14 epochs) 

Best Test accuracy - 99.21 

**Analysis:**

Over fitting in model 

Training accuracy would have been higher if trained for higher epochs 

Model is heavy for recognizing digits. Need to decrease parameters

**Log :**

![Alt text](images/code_1_log.PNG?raw=true "Optional Title")



**Code 2:**

**Target :**

Bringing the no.of parameters around 8k (required for late students)

 Introduction of GAP before output block to reduce parameters

**Results :**

Parameters - 8,024

Best Train accuracy - 98.59 (within 14 epochs)

Best Test accuracy - 98.44 

**Analysis:**

Less over fitting in model

Accuracy is less as the no.of parameters reduced a lot

Model performance can be improved by introducing regularization techniques

**Log :**



**Code 3:**

**Target :**

Introduced Regularization techniques like BN,dropout to improve performance

**Results :**

Parameters - 8,024

Best Train accuracy - 99.1 (within 14 epochs)

Best Test accuracy - 99.28 

**Analysis:**

Accuracy has improved with BN and dropout but less than 99.4%

Can use image augmentation techniques to correct for the slightly rotated images

**Log :**



**Code 4:**

**Target :**

Introduced image augmentation for higher accuracy

**Results :**

Parameters - 8,024

Best Train accuracy - 98.88 (within 14 epochs)

Best Test accuracy - 99.46

**Analysis:**

99.4% is maintained for last 4 of the 5 epochs

model is under fitting as training is made harder

No need to use LR technique as accuracy reached 99.4% within 14 epochs

**Log :**




