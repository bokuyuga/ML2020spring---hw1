# ML2020spring - hw1
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

最后，博主调整学习率为2，迭代6000次（多次炼丹的结果）  

除此之外，观察PM2.5的值，可以发现没有负数，都是整数，因此对预测值进行了微调，小于0的数都归为0，而对所有的浮点数四舍五入为整数  
