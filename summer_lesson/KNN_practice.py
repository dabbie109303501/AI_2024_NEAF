import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points


## first part : load dataset(very famous called iris)
iris=datasets.load_iris()
iris_data=iris.data
iris_label=iris.target


## second part : choose the label that we want (can be based on your preference)
column=[1,2]
iris_data=iris_data[:,column]

## third part : plot the distribution of data
for i in range(len(iris_data)):
    match iris_label[i]:
        case 0:
            plt.scatter(iris_data[i,0],iris_data[i,1],color="red",s=50,alpha=0.6)
        case 1:
            plt.scatter(iris_data[i,0],iris_data[i,1],color="green",s=50,alpha=0.6)
        case 2:
            plt.scatter(iris_data[i,0],iris_data[i,1],color="blue",s=50,alpha=0.6)
#plt.show()

## forth part : the principle of KNN
K=5
class_num=2
class_count=[0,0,0]
test_point=[3,2]
dis_array=[]
for i in range(len(iris_data)):
    dst=distance.euclidean(test_point,iris_data[i,:])
    dis_array.append(dst)

idx_sort=np.argsort(dis_array)[0:K]
for i in range(K):
    label=iris_label[idx_sort[i]]
    class_count[label]+=1
result=np.argsort(class_count)[-1]
print(result)

## fifth part : plot the test point

match result:
    case 0:
        plt.scatter(test_point[0],test_point[1],color="red",s=150,alpha=1,marker="^")
    case 1:
        plt.scatter(test_point[0],test_point[1],color="green",s=150,alpha=1,marker="^")
    case 2:
        plt.scatter(test_point[0],test_point[1],color="blue",s=150,alpha=1,marker="^")

plt.show()
