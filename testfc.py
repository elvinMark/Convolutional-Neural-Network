import neuralnetwork as nn 
import numpy as np 

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = [0,1,1,0]

mlp = nn.ConvNet()
mlp.addFCLayer(2,5,act_fun="relu")
mlp.addFCLayer(5,3,act_fun="relu")
mlp.addFCLayer(3,2,act_fun="relu")
mlp.addSoftmax()

mlp.train(x,y,maxIt=1000,alpha=0.1)
print(mlp.forward(x))