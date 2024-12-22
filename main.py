import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from resnet18 import ResNet18
from resnet import Bottleneck, ResNet, ResNet50, ResNet101

#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

net = ResNet50(10).to('cuda')
# net = ResNet18().to(device)
net_t = ResNet18().to(device)
net_t.load_state_dict(torch.load('best_resnet18_v2_state_dict.pth'))
net_t.eval()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

EPOCHS = 100
best_correct = 0
accuracy_list = []

for epoch in range(EPOCHS):
    losses = []
    kd_loss_arr = []
    running_loss = 0
    running_kd_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
    
        outputs = net(inputs)
        output_t = net_t(inputs)
        alpha = 3
        T = 8  # temperature
        kd_loss = 0.0
        kd_loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                            F.softmax(output_t/T, dim=1), reduction='batchmean') * alpha

        loss = criterion(outputs, labels)
        losses.append(loss.item())
        kd_loss_arr.append(kd_loss.item())
        # loss.backward()
        (loss + kd_loss).backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_kd_loss += kd_loss.item()
        if i%100 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0

            print(f'KD Loss [{epoch+1}, {i}](epoch, minibatch): ', running_kd_loss / 100)
            running_kd_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)

    avg_kd_loss = sum(kd_loss_arr)/len(kd_loss_arr)
    scheduler.step(avg_kd_loss)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if best_correct < correct:
        best_correct = correct
        torch.save(net.state_dict(), 'best_resnet50_kd_8_3_state_dict.pth')
        print('Model saved with accuracy: {:.2f}%'.format(100*(best_correct/total)))
        

    print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')
    accuracy_list.append(round(100*(correct/total), 2))
            
print('Training Done')
import csv
with open('test_resnet50_kd_8_3', 'w') as f:
     
    write = csv.writer(f)
    write.writerow(accuracy_list)
