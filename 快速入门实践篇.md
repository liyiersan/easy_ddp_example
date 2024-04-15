Pytorch DistributedDataParallel（DDP）教程二：快速入门实践篇

##### 一、简要回顾DDP

在上一篇文章中，简单介绍了Pytorch分布式训练的一些基础原理和基本概念。简要回顾如下：

1，DDP采用Ring-All-Reduce架构，其核心思想为：所有的GPU设备安排在一个逻辑环中，每个GPU应该有一个左邻和一个右邻，设备从它的左邻居接收数据，并将数据汇总后发送给右邻。通过N轮迭代以后，每个设备都拥有全局数据的计算结果。

2，DDP每个GPU对应一个进程，这些进程可以看作是相互独立的。除非我们自己手动实现，不然各个进程的数据都是不互通的。Pytorch只为我们实现了梯度同步。

3，DDP相关代码需要关注三个部分：数据拆分、IO操作、和评估测试。

##### 二、DDP训练框架的流程

###### 1. 准备DDP环境

在使用DDP训时，我们首先要初始化一下DDP环境，设置好通信后端，进程组这些。代码很简单，如下所示：

```python
def setup(rank, world_size):
    # 设置主机地址和端口号，这两个环境变量用于配置进程组通信的初始化。
    # MASTER_ADDR指定了负责协调初始化过程的主机地址，在这里设置为'localhost'，
    # 表示在单机多GPU的设置中所有的进程都将连接到本地机器上。
    os.environ['MASTER_ADDR'] = 'localhost'
    # MASTER_PORT指定了主机监听的端口号，用于进程间的通信。这里设置为'12355'。
    # 注意要选择一个未被使用的端口号来进行监听
    os.environ['MASTER_PORT'] = '12355'
    # 初始化分布式进程组。
    # 使用NCCL作为通信后端，这是NVIDIA GPUs优化的通信库，适用于高性能GPU之间的通信。
    # rank是当前进程在进程组中的编号，world_size是总进程数（GPU数量），即进程组的大小。
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 为每个进程设置GPU
    torch.cuda.set_device(rank)
```

###### 2. 准备数据加载器

假设我们已经定义好了dataset，这里只需要略加修改使用`DistributedSampler`即可。代码如下：

```python
def get_loader(trainset, testset, batch_size, rank, world_size):
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
	train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    # 对于测试集来说，可以选择使用DistributedSampler，也可以选择不使用，这里选择使用
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
	test_loader = DataLoader(test_set, batch_size=batch_size, sampler=train_sampler)
    # 不使用的代码很简单， 如下所示
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    rerturn train_loader, test_loader
```

注：关于testloader要不要使用分布式采样器，取决于自己的需求。如果测试数据集相对较小，或者不需要频繁进行测试评估，不使用`DistributedSampler`可能更简单，因为每个GPU或进程都会独立处理完整的数据集，从而简化了测试流程。然而，对于大型数据集，或当需要在训练过程中频繁进行模型评估的情况，使用`DistributedSampler`可以显著提高测试的效率，因为它允许每个GPU只处理数据的一个子集，从而减少了单个进程的负载并加快了处理速度。

对于DDP而言，每个进程上都会有一个dataloader，如果使用了`DistributedSampler`，那么真的批大小会是batch_size*num_gpus。

有关`DistributedSampler`的更多细节可以参考：

[DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)

###### 3. 准备DDP模型和优化器

在定义好模型之后，需要在所有进程中复制模型并用DDP封装。代码如下：

```python
def prepare_model_and_optimizer(model, rank, lr):
    # 设置设备为当前进程的GPU。这里`rank`代表当前进程的编号，
    # `cuda:{rank}` 指定模型应该运行在对应编号的GPU上。
    device = torch.device(f"cuda:{rank}") 
    # 包装模型以使用分布式数据并行。DDP将在多个进程间同步模型的参数，
    # 并且只有指定的`device_ids`中的GPU才会被使用。
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer
```

注：在DDP中不同进程之间只会同步梯度，因此为了保证训练时的参数同步，需要在训练开始前确保不同进程上模型和优化器的初始状态相同。对于优化器而言，当使用PyTorch中内置的优化器（如SGD, Adam等）时，只要模型在每个进程中初始化状态相同，优化器在每个进程中创建后的初始状态也将是相同的。但是，**如果是自定义的优化器，确保在设计时考虑到跨进程的一致性和同步，特别是当涉及到需要维护跨步骤状态（如动量、RMS等）时**。

确保模型的初始状态相同有如下两种方式：

1）参数初始化方法

在DDP中，每个GPU上都会有一个模型。我们可以利用统一的初始化方法，来保证不同GPU上的参数统一性。一个简单的示例代码如下：

```python
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
torch.manual_seed(42)  # 设置随机种子以确保可重复性
# 设置所有GPU的随机种子
torch.cuda.manual_seed_all(42)
model = MyModel()
model.apply(weights_init)
```

注：在初始化时，需要为所有的进程设置好相同的随机种子，不然weights_init的结果也会不一样。

2）加载相同的模型权重文件

另一种方法是在所有进程中加载相同的预训练权重。这确保了无论在哪个GPU上，模型的起点都是一致的。代码如下：

```python
model = MyModel()
model.load_state_dict(torch.load("path_to_weights.pth"))
```

注：如果你既没有设置初始化方法，也没有模型权重。一个可行的方式是手动同步，将rank=0的进程上模型文件临时保存，然后其他进程加载，最后再删掉临时文件。代码如下：

```python
def synchronize_model(model, rank, root='temp_model.pth'):
    if rank == 0:
        # 保存模型到文件
        torch.save(model.state_dict(), root)
    torch.distributed.barrier()  # 等待rank=0保存模型

    if rank != 0:
        # 加载模型权重
        model.load_state_dict(torch.load(root))
    torch.distributed.barrier()  # 确保所有进程都加载了模型

    if rank == 0:
        # 删除临时文件
        os.remove(root)
```

**模型同步似乎可以省略，在使用`torch.nn.parallel.DistributedDataParallel`封装模型时，它会在内部处理所需的同步操作**。

###### 4. 开始训练

训练时的代码，其实和单卡训练没有什么区别。最主要的就是在每个epoch开始的时候，要设置一下sampler的epoch，以保证每个epoch的采样数据的顺序都是不一样的。代码如下：

```python
def train(model, optimizer, criterion, rank, train_loader, num_epochs):
    sampler = train_loader.sampler
    for epoch in range(num_epochs):
        # 在每个epoch开始时更新sampler
        sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.cuda(rank), targets.cuda(rank)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # 只在rank为0的进程中打印信息
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
```

注：这里的打印的loss只是rank0上的loss，如果要打印所有卡上的平均loss，则需要使用all_reduce方法。代码如下：

```python
	# 将损失从所有进程中收集起来并求平均
    # 创建一个和loss相同的tensor，用于聚合操作
    reduced_loss = torch.tensor([loss.item()]).cuda(rank)
    # all_reduce操作默认是求和
    dist.all_reduce(reduced_loss)
    # 求平均
    reduced_loss = reduced_loss / dist.get_world_size()

    # 只在rank为0的进程中打印信息
    if rank == 0 and batch_idx % 100 == 0:
        print(f"Epoch {epoch}, Batch {batch_idx}, Average Loss: {reduced_loss.item()}")
```

###### 5. 评估测试

评估的代码也和单卡比较类似，唯一的区别就是，如果使用了`DistributedSampler`，在计算指标时，需要gather每个进程上的preds和gts，然后计算全局指标。

```python
def evaluate(model, test_loader, rank):
    model.eval()
    total_preds = []
    total_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(rank), targets.to(rank)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            # 收集当前进程的结果
            total_preds.append(preds)
            total_targets.append(targets)

    # 将所有进程的preds和targets转换为全局列表
    total_preds = torch.cat(total_preds).cpu()
    total_targets = torch.cat(total_targets).cpu()

    # 使用all_gather将所有进程的数据集中到一个列表中
    gathered_preds = [torch.zeros_like(total_preds) for _ in range(dist.get_world_size())]
    gathered_targets = [torch.zeros_like(total_targets) for _ in range(dist.get_world_size())]
    
    dist.all_gather(gathered_preds, total_preds)
    dist.all_gather(gathered_targets, total_targets)

    if rank == 0:
        # 只在一个进程中进行计算和输出
        gathered_preds = torch.cat(gathered_preds)
        gathered_targets = torch.cat(gathered_targets)
        
        # 计算全局性能指标
        accuracy = (gathered_preds == gathered_targets).float().mean()
        print(f'Global Accuracy: {accuracy.item()}')
```

注：如果test_loader没有设置`DistributedSampler`，评估的代码可以和单卡代码完全一样，不需要任何修改。

##### 三、完整代码

下面以CIFAR100数据集为例，完整展示一下DDP的训练流程。

```python
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

# 模型定义
class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # CIFAR100 has 100 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    # 销毁进程组
    dist.destroy_process_group()

def get_model():
    model = LeNet(100).cuda()
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    return model

def get_dataloader(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    rank = dist.get_rank()
    # 每个进程创建其独立的数据目录，避免I/O冲突
    # 这里使用rank来创建独立目录，例如：'./data_0'，'./data_1'等
    # 这种方法避免了多个进程同时写入同一个文件所导致的冲突
    # 注：这是一种简单的解决方案，但在需要大量磁盘空间的情况下并不高效，因为每个进程都需要存储数据集的一个完整副本。
    dataset = datasets.CIFAR100(root=f'./data_{rank}', train=train, download=True, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=train)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    return loader

def train(model, loader, optimizer, criterion, epoch, rank):
    model.train()
    # 设置DistributedSampler的epoch
    loader.sampler.set_epoch(epoch)
    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.cuda(rank), targets.cuda(rank)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 每100个batch计算当前的损失，并在所有进程中进行聚合然后打印
        if (batch_idx + 1) % 100 == 0:
            # 将当前的loss转换为tensor，并在所有进程间进行求和
            loss_tensor = torch.tensor([loss.item()]).cuda(rank)
            dist.all_reduce(loss_tensor)

            # 计算所有进程的平均损失
            mean_loss = loss_tensor.item() / dist.get_world_size()  # 平均损失

            # 如果是rank 0，则打印平均损失
            if rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx + 1}, Mean Loss: {mean_loss}")

def evaluate(model, dataloader, device):
    model.eval()
    local_preds = []
    local_targets = []

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            local_preds.append(preds)
            local_targets.append(targets)

    # 将本地预测和目标转换为全局列表
    local_preds = torch.cat(local_preds)
    local_targets = torch.cat(local_targets)

    # 使用all_gather收集所有进程的预测和目标
    world_size = dist.get_world_size()
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_targets, local_targets)
    
    # 只在rank 0进行计算和输出
    if dist.get_rank() == 0:
        gathered_preds = torch.cat(gathered_preds)
        gathered_targets = torch.cat(gathered_targets)
        accuracy = (gathered_preds == gathered_targets).float().mean()
        print(f"Global Test Accuracy: {accuracy.item()}")

def main_worker(rank, world_size, num_epochs):
    setup(rank, world_size)
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    start_time = time.time()
    for epoch in range(num_epochs):  # num of epochs
        train(model, train_loader, optimizer, criterion, epoch, rank)
        evaluate(model, test_loader, rank)
    # 计时结束前同步所有进程，确保所有进程已经完成训练
    dist.barrier()
    duration = time.time() - start_time
    
    if rank == 0:
        print(f"Training completed in {duration:.2f} seconds")
    cleanup()

if __name__ == "__main__":
    world_size = 4 # 4块GPU
    num_epochs = 10 # 总共训练10轮
    # 采用mp.spawn启动
    mp.spawn(main_worker, args=(world_size,num_epochs), nprocs=world_size, join=True)

```

注：

1）关于`get_loader`函数中数据加载有关部分的问题

```python
dataset = datasets.CIFAR100(root=f'./data_{rank}', train=train, download=True, transform=transform)
```

上面这段代码的最大问题在于，每个进程都会去下载一份数据到该进程对应的目录，这些目录之间是物理隔离的。显然，当要下载的数据集很大时，这种方法并不合适，因为会占用更多的硬盘资源，并且大量时间会花费在下载数据集上。但是如果不为每个进程设置单独的目录，就会造成读写冲突，多个进程都去同时读写同一个文件，最终导致数据集加载不成功。

一种更合理的解决方法是，提前下载好文件，并在创建数据集时设置download为False。代码如下：

```python
def download_data(rank):
    if rank == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 只在Rank 0 中下载数据集
        datasets.CIFAR100(root='./data_cifar100', train=True, download=True, transform=transform)
    # 等待rank0下载完成
    dist.barrier()
    
def get_dataloader(train=True):
    rank = dist.get_rank()
    # 现在只需要下载一次
    download_data(rank)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR100(root='./data_cifar100', train=train, download=False, transform=transform) # 设置download为False
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=train)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)
    return loader

```

2）关于DDP启动方式有关的问题

DDP的启动方式有两种，一种是**`torch.distributed.launch`**，这个工具脚本设置必要的环境变量，并为每个节点启动多个进程。通常在从命令行运行脚本时使用，在新版本的Pytorch 2.+版本中，这种方式是deprecated，不推荐继续使用。

还有一种就是**`torch.multiprocessing.spawn`：** 这个函数在Python脚本内部编程方式启动多个进程。启动方式很简单，分别传入主入口函数main_worker，然后传入main_woker的参数，以及GPU数量即可。

##### 四、DP, DDP性能对比

基于上述的代码，我还实现了一个DP的代码。实验setting为：

GPU: $4 \times$ RTX 4090, batch_size=256, optimizer为Adam，学习率为0.001，loss是CE loss。

它们之间的性能对比如下：

|   方式   | 时间  | 准确率 |
| :------: | :---: | :----: |
|   DDP    | 77秒  | 27.12% |
|    DP    | 293秒 | 26.76% |
| 单卡训练 | 248秒 | 26.34% |

没有调参，网络结构也是个最简单的LeNet，只训练了10轮，所以准确率比较低。不过，这个结果还是能说明一些问题的。可以看到DDP耗时最短，DP的时间反而比单卡训练还长。这主要是因为对于CIFAR100分类，单卡也可以很好地支持训练，显卡并不是性能瓶颈。当使用DP时，模型的所有参数在每次前向和反向传播后都需要在主GPU上进行聚合，然后再分发到各个GPU。这种多余的聚合和分发操作会带来显著的通信开销。并且在DataParallel中，主GPU承担了额外的数据分发和收集工作，会成为性能瓶颈，导致其他GPU在等待主GPU处理完成时出现闲置状态。

##### 五、总结

###### 1.  How to use DDP all depends on yourself. 

在最开始学习DDP的时候，有很多地方是很困惑的。每个博客的代码都有所区别，让我很是困惑。例如：在testloader到底要不要用`DistributedSampler`；在计算损失的时候，到底要不要用`all_reduce`操作来计算mean_loss；在计算指标的时候，到底要不要`all_gather`。后面了解多了之后才发现，到底用不用完全取决自己的需求。

1）mean_loss

由于Pytorch在DDP中会自动同步梯度，因此计算不计算mean_loss对于模型的训练和参数没有任何影响。唯一的区别在于打印日志的时候，是打印全局的平均损失，还是只打印某个进程上的损失。如果每张卡上的batch size已经足够大（例如，设置为128或者更高），打印全局平均损失和单进程上的损失，一般来说差别不大。

2）测试时`DistributedSampler`

测试时testloader设不设置`DistributedSampler`也完全取决于自己的实际需求。如果不设置，那么就是在每个进程上都会用全部的数据的来进行测试。如果有八块卡，那么就相当于在每个卡上都分别测试了一次，一共测试了八次。如果你的测试数据集比较小，比如只有几百张图像，并且测试的频率也不高的话，不设置`DistributedSampler`没有任何问题，不会有太多的额外开销。但是如果测试数据集比较大（比如几万张图像），并且训练时每个epoch都要进行测试，那么最好还是设置一下`DistributedSampler`，可以有效地减少总体训练时间。

3）评估时`all_gather`

至于要不要使用`all_gather`，则和有没有使用`DistributedSampler`相关。如果设置了`DistributedSampler`，那么评估时就要使用`all_gather`来汇总所有进程上的结果，否则打印的只会是某个进程的结果，并不准确。

4）batch_size

此外，testloader在使用`DistributedSampler`也需要格外注意数据能否被整除。举个例子，假设我们有8块卡，每块卡上的batch_size设置为64，那么总的batch size就是512。如果我们的训练数据集只有1000份，为了凑够完整的两个batch，`DistributedSampler`会对数据进行补全（重复部分数据）使得数据总数变为1024份。在这个过程有24份数据被重复评估，这些重复评估的数据可能会对评估结果产生影响。以4分类任务为例，如果类别数量比较均衡，相当于每个类别都有256份数据。在这种情况下，重复评估24份数据，对结果不会有什么影响。但是如果类别数据并不均衡，有些类别只有十几份数据，那么这个重复评估的影响就比较大了。如果正好重复的数据是样本数量只有十几份的类别，那么**评估结果将会变得极其不准确**！！！在这种情况下，我们需要重写一个sampler来实现无重复的数据采样。或者，也可以直接不使用`DistributedSampler`，在每个进程上都进行一次完整的评估。

5）同步批量归一化（Synchronized Batch Normalization, SynBN）

之前说过，每个GPU对应一个进程，不同进程的数据一般是不共享的。也就是说，如果模型结构中有BN层，每个GPU上的BN层只能访问到该GPU上的一部分数据，无法获得全局的数据分布信息。为此可以使用同步BN层，来利用所有GPU上的数据来计算BN层的mean和variance。代码也很简单，只需要对实例化model之后，转为同步BN即可。

```python
def get_model():
    model = LeNet(100).cuda()
    # 转换所有的BN层为同步BN层
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    return model
```

###### 2. 又Out了

以上是借助Pytorch提供的DDP有关API来搭建自己的分布式训练代码框架的教程，还是有一点小复杂的。现在有很多第三方库（如HuggingFace的[Accelerate](https://huggingface.co/docs/accelerate/en/index)、微软开发的[DeepSpeed](https://github.com/microsoft/DeepSpeed)、[Horovod](https://horovod.readthedocs.io/en/stable/)、[Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/)）对DDP进行了进一步的封装，使用这些第三方库可以极大地简化代码。但是，目前我还没有学习了解过这些第三方库（再次out了，没有及时学习前沿技术），有机会真应该好好学习一下。尤其是Horovod，它可以跨深度学习框架使用，支持Pytorch、TensorFlow、Keras、MXNet等。Accelerate和DeepSeed也很不错，做大模型相关基本上都会用到。Pytorch Lightning，顾名思义，让Pytorch变得更简单，确实Pytorch Lightning把细节封装得非常好，代码非常简洁，值得一学。

最后推荐两个B站上对DDP讲解很不错的几个视频：

[pytorch多GPU并行训练教程](https://www.bilibili.com/video/BV1yt4y1e7sZ/?spm_id_from=333.337.search-card.all.click&vd_source=27c7e890e929d2f5a77afaf77113e716)

[03 DDP 初步应用（Trainer，torchrun）](https://www.bilibili.com/video/BV13L411i7Ls/?spm_id_from=333.999.0.0)

知乎上有个帖子也不错：

[DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)