**Group - 25** - 

​		1) Ashok

​		2)Gokul



**Objective** - 

train a simple neural network using excel shet and understanding the concept of backward propagation and maths involved 

1)Explaining major steps - 



**Network Architecture** -
![Alt text](/Images/netwok.PNG?raw=true "Optional Title")

Network used - 2 layer network with 2 neurons in each. Each layer is a fully connected layer



No.of inputs - 2

No.of outputs - 2



**Forward propagation** - 

Using the initial random weights, outputs were predicted using the network. Then loss is calculated using the **mean squared error** loss function

​	layer 1 :

​			First layer weights - w1,w2,w3,w4

​			neurons - h1,h2

​			Activated neurons - a_h1,a_h2

​			h1 = w1*i1+w2*i2

​			h2 = w3*i1+w4*i2

​			Activation function - sigmoid

​			a_h1 = σ(h1)=1/(1+exp(-h1))

​			a_h2 = σ(h2)=1/(1+exp(-h2))

​	layer 2 :

​			Second layer weights - w5,w6,w7,w8

​			neurons - o1,o2

​			Activated neurons - a_o1,a_o2

​			o1 = w5*a_h1+w6*a_h2

​			o2 = w7*a_h1+w8*a_h2

​			Activation function - sigmoid

​			a_o1 = σ(o1) = 1/(1+exp(-o1))

​			a_o2 = σ(o2) = 1/(1+exp(-o2))

​	Loss calculation - 

​			Loss function - Mean Squared error

​			E1 = (1/2)*(t1-a_o1)^2

​			E2 = (1/2)*(t2-a_o2)^2

​			E_total = E1 + E2



**Backward propagation** - 

From the total loss, weights are updated through back propagation using **gradient descent**

**Updating the weight  of second layer**  - 

​	Updating the weight w5 - 

​	∂E_total/∂w5 =  ∂(E1+E2)/∂w5

​	∂E_total/∂w5 =  ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 *  ∂o1/w5

​	∂E1/∂a_o1 = (a_o1-t1)  -------------------------------------------------- 1

​	∂a_o1/∂o1 = σ(o1) *(1-σ(o1)) = a_o1*(1-a_o1)  ---------------------- 2

​	∂o1/w5 = a_h1 ------------------------------------------------------------- 3

​	from 1, 2 and 3 - 

​	**∂E_total/∂w5 =  (a_o1 - t1)*a_o1*(1 - a_o1)*a_h1**

​	similarly for w6,w7,w8 -

​	**∂E_total/∂w6 =  (a_o1 - t1)*a_o1*(1 - a_o1)*a_h2**

​	**∂E_total/∂w7 =  (a_o2 - t2)*a_o2*(1 - a_o2)*a_h1**

​	**∂E_total/∂w8 =  (a_o2 - t2)*a_o2*(1 - a_o2)*a_h2**

**Updating the weight  of first layer**  -

​	**∂E_total/∂w1 = [ (a_o1 - t1)*a_o1*(1 - a_o1)*w5 +   (a_o2 - t2)*a_o2*(1 - a_o2)*w7 ] * a_h1*(1 - a_h1) * i1**

​	**∂E_total/∂w2 = [ (a_o1 - t1)*a_o1*(1 - a_o1)*w5 +   (a_o2 - t2)*a_o2*(1 - a_o2)*w7 ] * a_h1*(1 - a_h1) * i2**

​	**∂E_total/∂w3 = [ (a_o1 - t1)*a_o1*(1 - a_o1)*w6 +    (a_o2 - t2)*a_o2*(1 - a_o2)*w8 ] * a_h2*(1 - a_h2) * i1**

​	**∂E_total/∂w4 = [ (a_o1 - t1)*a_o1*(1 - a_o1)*w6 +    (a_o2 - t2)*a_o2*(1 - a_o2)*w8] * a_h2*(1 - a_h2) * i2**

**Updating the Weights** - 

​	**wi = wi - (learning_rate)*∂E_total/∂wi**

