# ML2020spring - hw1&hw2  
# hw1:  
 Regression - PM2.5 Prediction  
Pesedo code:  
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

- 使用 Mini-batch gradient descent，使training data被分为许多mini-batches，每个小批次依次计算损失和梯度，并更新权重w和偏差b。一旦整个训练集训练完成，训练集会被打散并重新分配成小批次，进行下一轮计算。重复这个过程，直到达到设定的阈值。

- Loss的计算：cross_entropy_loss(y_pred, Y_label)  
    - cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))  
- 梯度的计算：gradient(X, Y_label, w, b)  
    - pred_error = Y_label - y_pred  
    - w_grad = -np.sum(pred_error * X.T, 1)     
    - b_grad = -np.sum(pred_error)  

Generative Model  
- 1.Find a function set  
    - 高斯分布有两个参数，miu和标准差，参数不同  
    - 或是选用別的distribution model  
    - 这些不同参数的的distribution集合起来，就是一个model  
- 2.Goodness of function  
    - Maximum Likelihood，越有可能sample出training data的分布model越好  
- 3.Find the best function  
    - 求最大likelihood，微分后得出最佳解  
