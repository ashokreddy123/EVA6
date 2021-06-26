### Problem Statement:

​	

​		i) Write a single model.py file that includes GN/LN/BN and takes an argument to  decide which      normalization to include, and then import this model file  to your notebook

​		ii) Normalization requirements - 

​				a)Network with Group Normalization + L1

​				b)Network with Layer Normalization + L2

​				c)Network with L1 + L2 + BN

​		iii) finding 20 misclassified images for each of the 3 models

​        iv)required graphs - 

​				a)Graph 1: Training Loss for all 3 models together

​				b)Graph 2: Test/Validation Loss for all 3 models together

​                c)Graph 3: Training Accuracy for all 3 models together

​                 d)Graph 4: Test/Validation Accuracy for all 3 models together



**Code Explanation:**



1) A model.py file is written with argument "**norm_arg**" to decide the type of normalization to be used.

2)  **norm_arg** = "**batch**" is for batch normalization. Loss function includes L1 and L2 loss

3)  **norm_arg** = "**layer**" is for layer normalization. Loss function includes L2 loss

4)  **norm_arg** = "**group**" is for group normalization. Loss function includes L1 loss

5)Batch Normalization is calculated for each channel of a layer over the batch. For batch normalization, no.of  means and std's of layer is equal to no.of its channels.

6)layer Normalization is calculated for each image of the batch. For layer normalization, no.of  means and std's  is equal to no.of images in the batch.

7)For group Normalization, for each image, certain no.of channels of a later are taken as a group. mean and std are calculated for this group. For group normalization, no.of  means and std's  is equal to no.of images in the batch*no.of groups for each image.

**Normalization calculation for batch size of 4  :**

![Alt text](images/2x2.PNG?raw=true "Optional Title")

**Graphs :**

1)**Training Loss:**

![Alt text](images/train_loss.PNG?raw=true "Optional Title")

2)**Testing Loss:**

![Alt text](images/test_loss.PNG?raw=true "Optional Title")

3)**Training Accuracy:**

![Alt text](images/train_accuracy.PNG?raw=true "Optional Title")

4)**Testing Accuracy:**

![Alt text](images/test_accuracy.PNG?raw=true "Optional Title")

**Misclassified Images  for:**

1)**Batch Normalization:**

![Alt text](images/batch_mis_classification.PNG?raw=true "Optional Title")

2)**Layer Normalization:**


![Alt text](images/layer_mis_classification.PNG?raw=true "Optional Title")

3)**Group Normalization:**


![Alt text](images/group_mis_classification.PNG?raw=true "Optional Title")

