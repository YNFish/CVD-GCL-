import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.data import DataLoader,Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

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

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight.float())
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight.float())
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight.float())
        z = self.encoder(x, edge_index, edge_weight.float())
        z1 = self.encoder(x1, edge_index1, edge_weight1.float())
        z2 = self.encoder(x2, edge_index2, edge_weight2.float())
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, f1

def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    # split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    # result = LREvaluator()(z, data.y, split)
    # return result
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = x.detach().cpu()
    x = torch.from_numpy(imp.fit_transform(x.numpy())).to('cuda')
    
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    accF1 = evaluate_model(x.cpu().numpy(), y.cpu().numpy())
    
    return result,accF1


def main():
    device = torch.device('cuda')
    
    repr = 'cpg'  # ast, cfg, pdg, cpg
    codetype = 'all'  # all, FFmpeg, qemu
    max_epoch = 2
    batch = 128
    graphdataset = f"graph2vec/{repr}/{codetype}_{repr}.pth"
    
    graphs,labels = torch.load(graphdataset)
    dataset = MyDataset(graphs, labels)
    # data = dataset.to(device)
    # print("dataset=",dataset)
    data = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))).to(device)
    # print("---------------------------")
    # print("data=",data)
    
    # dataloader = DataLoader(data, batch_size=batch)
    # input_dim = max(dataset.num_features, 1)
    # print(dataset.num_features)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, max_epoch+1):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result,accF1 = test(encoder_model, data)
    # print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
    print(f'{accF1[0]:.4f}  {accF1[1]:.4f}  {test_result["micro_f1"]:.4f}  {test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()