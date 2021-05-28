### Problem Statement:

​	Improve the given network to -

​		i) achieve 99.4% within 20 epochs

​		ii) use less than 20k parameters

​		iii) use BN, Dropout, a Fully connected layer and GAP

#### **Group - 25** -

​		1) Ashok

​		2)Gokul



#### **Coming up with Network Architecture:**

Since MNIST images are of small size(28*28),  first started with a receptive field of 7x7. But accuracy results are not good enough as receptive field is not high enough. So, it is increased to 9x9.

The number of convolutional blocks are increased from 1 to 3 to increase training accuracy.

To avoid loss of extracted features, 2D data is converted to 1D using Global Average Pooling(GAP) and not through Fully Connected Layer

After GAP, two FC layers are used. First is to increase the no.of channels and second is to output the array of size 10.

Dropout is not used as there is no overfitting

All convolutional kernels are of size 3x3 and Batch Normalization is used after every layer



#### **Loss Function:**

​	Since, it is multi-class classification, nll loss function is used on log_softmax of output

#### Number of parameters

​	The total number of parameters are - **15,262**

#### **Test Accuracy**  

​	After 20 epochs, test accuracy - **99.18 %**

#### **Training log** 



#### **Results**