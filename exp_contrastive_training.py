from collections import defaultdict
from datasets.utils.collate import Collater
from datasets.functions_dataset import FunctionsDataset
import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d as BN, ModuleList, Embedding, Identity
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool


class GIN0(Module):
    def __init__(self, node_features: int, hidden_dim: int, num_layers: int = 5, train_eps: bool = False):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(node_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=train_eps)
        self.convs = ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        BN(hidden_dim),
                    ), train_eps=train_eps))
        self.lin1 = Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = torch.tanh(self.lin1(x))
        return x

    def __repr__(self):
        return self.__class__.__name__


class GraphEncoder(Module):
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 num_layers: int = 5,
                 node_labels: Optional[int] = None,
                 node_embeddings: Optional[int] = None,
                 train_eps: bool = False):
        super().__init__()
        self.out_dim = hidden_dim

        if node_embeddings is not None:
            assert node_labels is not None, "Pass node_labels if you want to use node embeddings"
            self.node_embeddings = Embedding(node_labels, embedding_dim=node_embeddings)
        else:
            self.node_embeddings = Identity()

        node_features = node_embeddings if node_embeddings is not None else input_dim
        self.model = GIN0(node_features, hidden_dim, num_layers, train_eps)


    def forward(self, data: Data) -> torch.FloatTensor:
        data.x = self.node_embeddings(data.x).squeeze().float()
        return self.model(data)


class GraphClassifier(Module):

    def __init__(self, encoder, num_classes):
        super().__init__()
        self.classifier = Linear(encoder.out_dim, num_classes)
        self.encoder = encoder

    def forward(self, x):
        return self.classifier(self.encoder(x))


class SiameseClassifier(Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = Linear(encoder.out_dim, 2)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        diff = x1 - x2
        return torch.sigmoid(self.head(diff))


class Net(Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = BN(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = BN(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = BN(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = BN(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = BN(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)



class PairDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.label2index = defaultdict(list)
        for i in range(len(dataset)):
            l = dataset[i].y
            self.label2index[l].append(i)
        self.pos = True
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        g1 = self.dataset[index]
        if self.pos:
            index2 = random.choice(self.label2index[g1.y])
        else:
            indexes = []
            for k, v in self.label2index.items():
                if k == g1.y:
                    continue
                indexes.extend(v)
            index2 = random.choice(indexes)
        g2 = self.dataset[index2]

        self.pos = not self.pos

        return g1, g2, g1.y == g2.y


def collater(data_list):
    batch_1 = Batch.from_data_list([d[0] for d in data_list])
    batch_2 = Batch.from_data_list([d[1] for d in data_list])
    labels = torch.tensor(data_list[2])
    return batch_1, batch_2, labels



class ENZYMES():
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # self.transform = T.Compose([
        #     T.OneHotDegree(self.num_features - 1),
        #     T.ToSparseTensor(),
        # ])

    def prepare_data(self):
        TUDataset(self.data_dir, name='ENZYMES')
                #   pre_transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        dataset = TUDataset(self.data_dir, name='ENZYMES')
        self.num_classes = dataset.num_classes
        self.node_features = dataset.num_node_features
        dataset = dataset.shuffle()
        self.test_dataset = dataset[:len(dataset) // 10]
        self.val_dataset = dataset[len(dataset) // 10:len(dataset) // 5]
        self.train_dataset = dataset[len(dataset) // 5:]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128)

# dataset = TUDataset("tmp/MUTAG", name='MUTAG')
# num_classes = dataset.num_classes
# node_features = dataset.num_node_features
# test_dataset = dataset[:len(dataset) // 10]
# train_dataset = dataset[len(dataset) // 10:]
# train_dataset = PairDataset(train_dataset)
# test_dataset = PairDataset(test_dataset)
# test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collater)
# train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collater)

# enzymes = ENZYMES("tmp")
# enzymes.setup()
# node_features, num_classes = enzymes.node_features, enzymes.num_classes
# train_dataset = enzymes.train_dataset
# train_loader = enzymes.train_dataloader()
# test_loader = enzymes.test_dataloader()

train_dataset = FunctionsDataset("data/graph_functions", "train.txt", "pairs", "graph", 1000, "data/cache", only_augs=False, remove_self_loops=True)
test_dataset = FunctionsDataset("data/graph_functions", "train.txt", "pairs", "graph", 10, "data/cache", only_augs=False, remove_self_loops=True)
node_features = 1
node_labels = train_dataset.num_tokens
embs = 4
train_loader = DataLoader(train_dataset, num_workers=6, batch_size=128, collate_fn=Collater())
test_loader = DataLoader(test_dataset, num_workers=6, batch_size=128, collate_fn=Collater())




device = torch.device('cuda')
# model = GraphClassifier(GraphEncoder(input_dim=node_features), num_classes).to(device)
# model = Net(node_features, num_classes).to(device)
model = SiameseClassifier(GraphEncoder(input_dim=node_features, node_labels=node_labels, node_embeddings=embs)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = 0
    for d1, d2, l1, l2 in train_loader:
        d1 = d1.to(device)
        d2 = d2.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        ls = (l1 == l2).long()
        optimizer.zero_grad()
        out = model(d1, d2)
        loss = F.cross_entropy(out, ls)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for i in range(100):
        for d1, d2, l1, l2 in loader:
            d1 = d1.to(device)
            d2 = d2.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            ls = (l1 == l2).long()
            pred = model(d1, d2)
            pred = torch.argmax(pred, 1)
            correct += (pred == ls).sum().item()
    return correct / (100 * len(loader.dataset))


for epoch in range(1, 201):
    loss = 0.
    for i in range(10):
        loss += train()
    test_acc = test(test_loader)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))
    scheduler.step()