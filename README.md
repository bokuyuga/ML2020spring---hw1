# ML2020spring - hw1&hw2  
# hw1:  
 Regression - PM2.5 Prediction
 - Pesedo code:  
     -  Declare weight vector, initial Ir, and # of iteration  
     -  for i_th iteration:  
         - y'=the inner product of train_x and weight vector  
         - Loss=y'-train_y  
         - gradient=2*np.dot((train_x)',Loss)  
         - prev_gra += gra**2  
         - ada = np.sqrt(prev_gra)  
         - weight vector-=learning rate*gradient  
           
Adagrad更新公式：  

Adam更新公式：  

最后，调整学习率为2，迭代6000次 

除此之外，观察PM2.5的值，可以发现没有负数，都是整数，因此对预测值进行了微调，小于0的数都归为0，而对所有的浮点数四舍五入为整数  


# hw2:
 Classification -  Census-Income (KDD) Data Set
 - build linear binary classifiers to predict whether the income of an indivisual exceeds 50,000 or not.  
 - logistic regression(LR)  
 - linear discriminant anaysis(LDA)  
 (compare the differences between the two, or explore more methodologies)

- 使用 Mini-batch gradient descent，使training data被分为许多mini-batches，每个小批次依次计算损失和梯度，并更新权重w和偏差b。一旦整个训练集训练完成，训练集会被打散并重新分配成小批次，进行下一轮计算。We repeat such process until max number of iterations is reached.
