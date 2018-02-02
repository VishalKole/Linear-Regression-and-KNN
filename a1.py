from pylab import *

data = load("data.npy")
#
# i=0
# while(i<len(data[:,1])-1):
#     i=i+1
#     if(data[i,2]==0):
#         plt.scatter(data[i, 0], data[i, 1], marker='o', color='b')
#     else:
#         plt.scatter(data[i,0],data[i,1],marker='o',color='g')
#
# v = axis()
# gap=0.1
#
# for x in np.arange(v[0],v[1], gap):
#     for y in np.arange(v[2],v[3], gap):
#         plt.scatter(x,y,marker='o',color='r' ,s=0.2)
#
x=[]
x = data[: ,0:2]

y = x.transpose()

zd = np.matmul(x,y)

zf= inv(zd)

zg= np.matmul(zf,x)

zk= data[:,2]

#zk= zk.transpose()

zr = np.matmul(zk, zg)

plt.show()