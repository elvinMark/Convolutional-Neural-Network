import numpy as np 
import cv2 as cv 
import json

def conv(I,F,backward=False):
	ishape = I.shape
	fshape = F.shape
	if backward:
		xsize = ishape[0] + fshape[0] - 1
		ysize = ishape[1] + fshape[1] - 1
		out = np.zeros([xsize,ysize])
		for i in range(ishape[0]):
			for j in range(ishape[1]):
				out[i:i+fshape[0],j:j+fshape[1]] = I[i][j]*F
		return out
	xsize = ishape[0] - fshape[0] + 1
	ysize = ishape[1] - fshape[1] + 1
	out = np.zeros([xsize,ysize])
	for i in range(xsize):
		for j in range(ysize):
			out[i][j] = sum(sum(F*I[i:i+fshape[0],j:j+fshape[1]]))
	return out

def pool(I,order,ptype,backward=False):
	ishape = I.shape
	if backward:
		out = np.zeros([ishape[0]*order,ishape[1]*order])
		ones = np.ones([order,order])
		for i in range(ishape[0]):
			for j in range(ishape[1]):
				out[order*i:order*(i+1),order*j:order*(j+1)] = I[i][j]*ones
		return out	
	
	out = np.zeros([ishape[0]//order,ishape[1]//order])
	if ptype=="average" :
		for i in range(ishape[0]//order):
			for j in range(ishape[1]//order):
				out[i][j] = sum(sum(I[order*i:order*(i+1),order*j:order*(j+1)]))/(order*order)
	else:
		for i in range(ishape[0]//order):
			for j in range(ishape[1]//order):
				out[i][j] = max(I[order*i:order*(i+1),order*j:order*(j+1)])
	return out

def sigmoid(x,diff):
	if diff:
		return x*(1-x)
	return 1/(1+np.exp(-x))
def tanh(x,diff):
	if diff:
		return (1-x**2)/2
	return (1-np.exp(-x))/(1+np.exp(-x))
def relu(x,diff):
	if diff:
		return np.where(x>=0,1,0.1)
	return np.where(x>=0,x,0.1*x)
def linear(x,diff):
	if diff:
		return np.ones(x.shape)
	return x
class FCLayer:
	def __init__(self,nin,nout,act_fun = "sigmoid"):
		self.w = np.random.random([nin,nout])#Creating the weight matrix with random values
		self.bias = np.random.random([1,nout])#Create the bias vector (kind of the threshold)
		self.ones_matrix = None
		self.nin = nin
		self.nout = nout
		self.act_fun = act_fun
		self.in_data = None
		self.out_data = None
		self.delta = None
	def activation_function(self,x,diff=False):
		if self.act_fun == "sigmoid":
			return sigmoid(x,diff)
		elif self.act_fun == "tanh":
			return tanh(x,diff)
		elif self.act_fun == "relu":
			return relu(x,diff)
		elif self.act_fun == "linear":
			return linear(x,diff)
		else:
			return sigmoid(x,diff)
	def forward(self,in_data):
		self.in_data = in_data
		self.ones_matrix = np.ones([len(in_data),1])
		o = self.in_data.dot(self.w) + self.ones_matrix.dot(self.bias)
		self.out_data = self.activation_function(o,diff=False)
		return self.out_data
	def backward(self,err):
		self.delta = self.activation_function(self.out_data,diff=True)*err
		return self.delta.dot(self.w.T)
	def update(self,alpha):
		self.w = self.w - alpha*self.in_data.T.dot(self.delta)
		self.bias = self.bias - alpha*np.sum(self.delta,axis=0,keepdims=True)
	def toJSON(self):
		s = {}
		s["w"] = self.w.tolist()
		s["bias"] = self.bias.tolist()
		s["nin"] = self.nin
		s["nout"] = self.nout
		s["act_fun"] = self.act_fun
		#return json.dumps(s)
		return {"type":"FC","data":s}
	def fromJSON(self,s):
		#data = json.loads(s)
		data = s
		self.w = np.array(data['w'])
		self.bias = np.array(data['bias'])
		self.nin = data['nin']
		self.nout = data['nout']
		self.act_fun = data['act_fun']

class ConvLayer:
	def __init__(self,fsize,nchannels):
		self.fsize = fsize
		self.nchannels = nchannels
		self.filters = []
		self.o = []
		self.i = []
		for i in range(nchannels):
			self.filters.append(np.random.random(fsize)/(fsize[0]*fsize[1]))
		self.filters = np.array(self.filters)
	def forward(self,Iset):
		self.o = []
		self.i = []
		for img in Iset:
			self.i.append(img)
			for f in self.filters:
				self.o.append(conv(img,f))
		self.o = np.array(self.o)
		return self.o
	def backward(self,err):
		self.e = []
		self.d = err
		for img in err:
			tmp = []
			for f in self.filters:
				if len(tmp)!=0:
					tmp = tmp + conv(img,f,backward=True)
				else:
					tmp = conv(img,f,backward=True)
			self.e.append(tmp)
		self.e = np.array(self.e)
		return self.e
	def update(self,alpha):
		counter = 0
		for img in self.i:
			for k in range(len(self.filters)):
				for delta in self.d:
					h,w = delta.shape
					for x in range(h):
						for y in range(w):
							self.filters[k] -= img[x:x+self.fsize[0],y:y+self.fsize[1]]*delta[x][y];
	def toJSON(self):
		a = {}
		a["type"] = "Conv"
		ed = []
		for i in self.filters:
			ed.append(i.tolist())
		a["data"] = {"fsize":self.fsize,"nchannels":self.nchannels,"filters":ed}
		return a
	def fromJSON(self,data):
		self.fsize = data["fsize"]
		self.nchannels = data["nchannels"]
		self.filters = []
		for f in data["filters"]:
			self.filters.append(np.array(f))
		self.filters = np.array(self.filters)

class PoolLayer:
	def __init__(self,order,ptype="average"):
		self.o = []
		self.order=order
		self.ptype = ptype
	def forward(self,Iset):
		self.o = []
		for img in Iset:
			self.o.append(pool(img,self.order,self.ptype))
		self.o = np.array(self.o)
		return self.o
	def backward(self,err):
		self.e = []
		for img in err:
			self.e.append(pool(img,self.order,self.ptype,backward=True))
		self.e = np.array(self.e)
		return self.e
	def update(self,alpha):
		pass
	def toJSON(self):
		a = {}
		a["type"] = "Pool"
		a["data"] = {"order":self.order,"ptype":self.ptype}
		return a
	def fromJSON(self,data):
		self.order = data["order"]
		self.ptype = data["ptype"]

class FixingLayer:
	def __init__(self,insize,outsize):
		self.insize = insize
		self.outsize = outsize
	def forward(self,Iset):
		return Iset.reshape(self.outsize)
	def backward(self,Iset):
		return Iset.reshape(self.insize)
	def update(self,alpha):
		pass
	def toJSON(self):
		a = {}
		a["type"] = "Fixing"
		a["data"] = {"insize" : self.insize,"outsize" : self.outsize}
		return a
	def fromJSON(self,data):
		self.insize = data["insize"]
		self.outsize = data["outsize"]

class Softmax:
	def __init__(self):
		self.sm = None
	def forward(self,indata):
		self.sm = []
		for elem in indata:
			s = np.exp(elem)
			self.sm.append(s/sum(s))
		self.sm = np.array(self.sm)
		return self.sm
	def backward(self,target):
		err = []
		for elem,t in zip(self.sm,target):
			elem[t] -= 1
			err.append(elem)
		return np.array(err)
	def update(self,alpha):
		pass

class ConvNet:
	def __init__(self):
		self.layers = []
		self.softmax = False
	def addConvLayer(self,fsize,nchannels):
		self.layers.append(ConvLayer(fsize,nchannels))
	def addPoolLayer(self,order,ptype="average"):
		self.layers.append(PoolLayer(order,ptype=ptype))
	def addFixingLayer(self,insize,outsize):
		self.layers.append(FixingLayer(insize,outsize))
	def addFCLayer(self,nin,nout,act_fun="tanh"):
		self.layers.append(FCLayer(nin,nout,act_fun))
	def addSoftmax(self):
		self.layers.append(Softmax())
		self.softmax = True
	def forward(self,I):
		o = I
		for l in self.layers:
			o = l.forward(o)
		return o
	def backward(self,err):
		e = err
		for l in reversed(self.layers):
			e = l.backward(e)
	def update(self,alpha):
		for l in self.layers:
			l.update(alpha)
	def train(self,indata,outdata,maxIt=100,alpha=0.1):
		for i in range(maxIt):
			o = self.forward(indata)
			if(self.softmax):
				self.backward(outdata)
			else:
				e = o-outdata
				self.backward(e);
			self.update(alpha)
	def save(self,dirname="convnet.json"):
		f = open(dirname,"w")
		s = {}
		counter = 0
		for l in self.layers:
			s["l"+str(counter)] = l.toJSON()
			counter+=1 
		f.write(json.dumps(s))
		f.close()
	def load(self,dirname="convnet.json"):
		f = open(dirname)
		j = json.load(f)
		self.layers = []
		for l in j:
			a = j[l]
			if a["type"] == "FC":
				tmp = FCLayer(1,1)
			elif a["type"] == "Pool":
				tmp = PoolLayer(1)
			elif a["type"] == "Fixing":
				tmp = FixingLayer(1,1)
			elif a["type"] == "Conv":
				tmp = ConvLayer([1,1],1)
			tmp.fromJSON(a["data"])
			self.layers.append(tmp)
		f.close()

if __name__=="__main__":
	cn = ConvNet()

	test = np.random.random([20,20]).reshape(1,20,20)/100
	testo = np.random.random([1,3])
	cn.addConvLayer([5,5],5)#16x16
	cn.addPoolLayer(2)#8x8
	cn.addConvLayer([5,5],2)#4x4
	cn.addPoolLayer(4)#11x11
	cn.addFixingLayer([1,1,10],[1,10])
	cn.addFCLayer(10,3)

	out = cn.forward(test)
	print(out)
	cn.backward(out-testo)
	"""
	Image
	test = np.load("data/test-data-0.dat.npy")
	test = test.reshape(1,200,200)/40000
	cn.addConvLayer([5,5],5)#196x196
	cn.addPoolLayer(4)#49x49
	cn.addConvLayer([6,6],2)#44x44
	cn.addPoolLayer(4)#11x11
	cn.addConvLayer([4,4],2)#8x8
	cn.addPoolLayer(8)
	cn.addFixingLayer([1,1,20],[1,20])
	cn.addFCLayer(20,3)
	print(cn.forward(test))
	"""

			
