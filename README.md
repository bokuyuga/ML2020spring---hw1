# ML2020spring - hw1
 Regression - PM2.5 Prediction
 Pesedo code:  
 Declare weight vector, initial Ir, and # of iteration  
   for i_th iteration:  
     y'=the inner product of train_x and weight vector  
     Loss=y'-train_y  
     gradient=2*np.dot((train_x)',Loss)  
     prev_gra += gra*2  
     ada = np.sqrt(prev_gra)  
     weight vector-=learning rate*gradient  
