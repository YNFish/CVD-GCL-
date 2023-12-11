import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import numpy as np

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
        self.num_features = max(graph.ndata['code'].shape[1] + graph.ndata['operator'].shape[1] + 1 for graph in graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        x = torch.cat([graph.ndata['code'], graph.ndata['operator'], graph.ndata['subscript'].unsqueeze(-1)], dim=-1)
        data = Data(x=x, edge_index=torch.stack(graph.edges()), edge_attr=graph.edata['repr'], y=torch.tensor(label))
        return data


def make_gin_conv(input_dim, out_dim):
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        # print("------------------------------------- batch ------------------------------------")
        # print(batch)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        # print(data)
        data = data.to('cuda')
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def evaluate_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, f1


def test(encoder_model, dataloader):
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
# set    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    # split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # x = imp.fit_transform(x.detach().cpu().numpy())
    # result = SVMEvaluator(linear=True)(x, y, split)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = x.detach().cpu()
    x = torch.from_numpy(imp.fit_transform(x.numpy())).to('cuda')
    
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    accF1 = evaluate_model(x.cpu().numpy(), y.cpu().numpy())
    
    return result,accF1


def deleteErrorData(data):
    data = data.cpu()
    # 检查每个数组的形状
    for key, value in data.items():
        print(f"{key}: {np.array(value).shape}")

    # 识别需要删除的数据的条件，这里假设删除 x 数组为空的情况
    rows_to_delete = np.where(np.array(data['x']).shape[1] == 0)[0]

    # 删除数据
    for key, value in data.items():
        data[key] = np.delete(value, rows_to_delete, axis=0)

    # 打印删除后的形状
    for key, value in data.items():
        print(f"{key}: {np.array(value).shape}")
        
    return data.cuda()


def main():
    device = torch.device('cuda')
    
    # 只需要修改这个
    repr = 'cpg'  # ast, cfg, pdg, cpg
    codetype = 'all'  # all, FFmpeg, qemu
    max_epoch = 2
    batch = 256
    graphdataset = f"graph2vec/{repr}/{codetype}_{repr}.pth" 
    
    graphs,labels = torch.load(graphdataset)
    dataset = MyDataset(graphs, labels)
    # print("dataset=",dataset)
    data = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))).to(device)
    # print("---------------------------")
    # print("data=",data)
    
    # data = deleteErrorData(data)
    
    # print("after data=",data)
        
    dataloader = DataLoader(data, batch_size=batch)
    input_dim = max(dataset.num_features, 1)
    print(dataset.num_features)

    # aug = A.RWSampling()
    # aug1 = A.FeatureMasking(pf=0.1)
    # aug2 = A.FeatureDropout(pf=0.2)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')
print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')

    
    with tqdm(total=max_epoch, desc='(T)') as pbar:
        for epoch in range(1, max_epoch+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result,accF1 = test(encoder_model, dataloader)
    print(accF1)
    print(test_result)
    # print(type(test_result))
    # print("Accuracy:" ,test_result['accuracy'])
    # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    print(graphdataset)
    print(f'epoch={max_epoch}')
    print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()