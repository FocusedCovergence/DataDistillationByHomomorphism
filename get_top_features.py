import torch
import torch.nn.functional as F

def get_top_n_features(model, data, n, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    if data.x.grad is None:
        data.x.requires_grad = True
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

    gradients = data.x.grad
    feature_importance = torch.sum(torch.abs(gradients), dim=0)
    top_n_indices = torch.topk(feature_importance, n).indices.cpu().numpy()
    return top_n_indices