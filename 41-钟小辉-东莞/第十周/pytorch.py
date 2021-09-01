import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Model:
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass
    def create_cost(self,cost):
        support_cost = {'CROSS_ENTROPY': nn.CrossEntropyLoss(),
                        'MSE':nn.MSELoss()}
        return support_cost[cost]
    def create_optimizer(self,optimist,**rests):
        support_optimist = {'SGD':optim.SGD(self.net.parameters(),lr =0.1,momentum=0.9,
                          weight_decay=5e-4,**rests),
                            'ADAM':optim.Adam(self.net.parameters(),lr=0.1,**rests),
                            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.1,**rests)}

        return support_optimist[optimist]

    def train(self,train_loader,epoches = 3):
        for epoch in range(epoches):
            running_lost = 0.0
            for i,data in enumerate(train_loader,0):
                inputs,labels = data
                self.optimizer.zero_grad()

                #forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()

                running_lost += loss.item()
                if i % 100 == 0:
                    print('[epoch %d,%.2f%%] loss:%.3f'%
                          (epoch+1,(i+1)*1./len(train_loader),running_lost/100))
                    running_lost = 0.0

        print("Finished Training")

    def evaluate(self,test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad(): # no grad when test and predict
            for data in test_loader:
                images,labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images:%d %%'% (100*correct/total))

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        #输入32 ，1， 28，28
        self.conv1 = nn.Conv2d(1, 56, 3, padding=1)  # 56*28*28
        self.conv2 = nn.Conv2d(56, 56, 3, padding=1)  # 56*28*28
        self.pool1 = nn.MaxPool2d(2,2) # 56*16*16
        self.bn1 = nn.BatchNorm2d(56)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(56,112, 3, padding=1)  # 112*14*14
        self.conv4 = nn.Conv2d(112, 112, 3, padding=1)  # 112*14*14
        self.pool2 = nn.MaxPool2d(2,2)   # 112*7*7
        self.bn2 = nn.BatchNorm2d(112)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(112,224, 3, padding=1)  # 224*7*7
        self.conv6 = nn.Conv2d(224, 224, 3, padding=1)  # 224*7*7
        self.pool3 = nn.MaxPool2d(2,2,padding=1)   # 224*4*4
        self.bn3 = nn.BatchNorm2d(224)
        self.relu3 = nn.ReLU()

        self.conv7 = nn.Conv2d(224, 448, 3, padding=1)  # 448*4*4
        self.conv8 = nn.Conv2d(448, 448, 3, padding=1)  # 448*4*4
        self.pool4 = nn.MaxPool2d(2, 2)  # 448*2*2
        self.bn4 = nn.BatchNorm2d(448)  # 448*2*2
        self.relu4 = nn.ReLU()  # 448*2*2


        # self.conv1 = torch.nn.Conv2d(1,)
        self.fc1 = torch.nn.Linear(448*2*2,512)
        self.drop1 = torch.nn.Dropout2d()
        self.fc2 = torch.nn.Linear(512,512)
        self.drop2 = torch.nn.Dropout2d()
        self.fc3 = torch.nn.Linear(512,10)

    def forward(self,x):
        #输入为32*28*28 c,h,w
        # x = x.view(-1,28,28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)


        x = x.view(-1,448*2*2) #reshape

        # print("x_shape:", x.size())
        x = F.relu(self.fc1(x))
        self.drop1(x)
        x = F.relu(self.fc2(x))
        self.drop2(x)
        x = F.softmax(self.fc3(x),dim=1)

        return x


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])])
    trainset  = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

if __name__ == "__main__":
    net = Network()
    model = Model(net ,"CROSS_ENTROPY","SGD")
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)


