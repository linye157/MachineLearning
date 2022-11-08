import torch
import torch.nn as nn
import pandas as pd

#负责训练数据和训练测试数据，读取像素和对应标签
class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        item = self.data.iloc[index]
        label,features = item[0],torch.Tensor(item[1:])
        return features,label

def main():
    #超参数的设置
    input_size = 784 #输入的大小28*28
    num_classes = 10 #输出分类的大小
    num_epochs = 5 #迭代的次数5
    batch_size = 30 #一次学习的数据量30
    learning_rate = 0.001 #学习率

#为训练和测试任务创建的数据集
    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')
    train_dataset = DatasetMNIST(data_train)
    test_dataset = DatasetMNIST(data_test)


#数据加载器，用来帮助读取训练和测试所使用的的数据，同时设置批次大小和数据乱序
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

#构建一个线性模型
    model = nn.Linear(input_size, num_classes)
#损失函数，交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
#优化器，使用随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#训练模型
    total_step = len(train_loader)
#迭代nun——epochs次
    for epoch in range(num_epochs):
        for i,(features,labels) in enumerate(train_loader):
            #前项传播
            outputs = model(features)
            #计算损失
            loss = criterion(outputs,labels)

            #反向传播及优化过程
            #清空梯度缓存，避免影响到下一个batch
            optimizer.zero_grad()
            #反向传播，计算新的梯度
            loss.backward()
            #更新梯度
            optimizer.step()


            if(i + 1) % 10 == 0:
                print('Epoch[{}/{}],Step,Loss{:.4f}'.format(epoch + 1,num_epochs,i + 1,total_step,loss.item()))




#模型测试
    with torch.no_grad():
        correct = 0
        total = 0
        for features,labels in test_loader:
           #预测输出
            outputs = model(features)
           #_代表最大值，predicted代表最大值所在index，1为输出所在列最大值
            _,predicted = torch.max(outputs.data,1)
           #返回labels的个数
            total += labels.size(0)
           #计算正确预测个数
            correct += (predicted == labels).sum()

        print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))
        #保存训练好的模型
    torch.save(model.state_dict(),"model.ckpt")


if __name__ == '__main__':
    main()
    # for i, features in enumerate(test_dataset):
    #     outputs = model(features)  # bsz,10
    #     _, predicted = torch.max(outputs, 0)
    #     res.append([i + 1, predicted.item()])
    # dataframe = pd.DataFrame(res, columns=['ImageId', 'Label'])
    # dataframe.to_csv("sample_submission.csv", index=False)
