import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlrd

# 颜色列表-2
color_List = ['yellow','red','green','purple','black']

# 分类函数
def classify(data,centers):
    length = centers.shape[0]  #查询有多少个中心点
    classes = [ [] for i in range(length) ]
    sumDis = 0

    for i in range(data.shape[0]) :
        per_data = data[i]
        diffMat = np.tile(per_data,(length,1))-centers  #复制并计算距离
        sqDiffMat = diffMat**2
        sqDisMat = sqDiffMat.sum(axis=1) #按行求和
        sortedIndex = sqDisMat.argsort() #对距离进行排序并得到相应指令
        classes[sortedIndex[0]].append(list(per_data))
        sumDis += sqDisMat[sortedIndex[0]]
    return classes,sumDis

# 中心点更新
def upCenters(classes):
    centers = []
    for i in range(len(classes)) :
        per_class = classes[i]
        per_class = np.array(per_class)
        center = per_class.sum(axis=0)/len(per_class) #按列求和求中心点
        centers.append(center)
    return np.array(centers)

# 聚类函数
def kmeans(data,centers,sumDis) :
    #聚类流程：聚类——修改中心点——再聚类——再修改中心点——两次聚类的总距离一样则戒指
    # 聚类
    classes,new_sumDis = classify(data, centers)
    if sumDis == new_sumDis :
        return
    # 修改中心
    new_centers = upCenters(classes)

    # 绘图
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    for i in range(len(new_centers)):
        center = centers[i]
        ax.scatter(center[0], center[1], center[2], marker='x', c=color_List[i])

    for i in range(len(classes)):
        per_class = classes[i]
        for c in per_class:
            ax.scatter(c[0], c[1], c[2], c=color_List[i])

    plt.title('sumDis%f'%new_sumDis)
    plt.show()

    kmeans(data,new_centers,new_sumDis)

# 提取数组信息
def getdata(xls) :
    workbook = xlrd.open_workbook(xls)
    worksheet = workbook.sheet_by_index(0)
    nrows,ncols = worksheet.nrows,worksheet.ncols
    data = []

    for i in range(nrows) :
        temp = []
        for j in range(ncols):
            temp.append(worksheet.cell_value(i,j))
        data.append(temp)

    return np.array(data)  #转化成数组

if __name__ == '__main__':
    data = getdata('data.xls')
    # print(data)
    centers = data[:3]
    kmeans(data,centers,0)


