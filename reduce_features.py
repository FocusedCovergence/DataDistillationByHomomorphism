import torch
from torch_geometric.data import Data

def reduce_features(df_features, top_m_categorical_features, data):
    df_new_features = df_features[top_m_categorical_features]
    new_features = torch.tensor(df_new_features.values, dtype=torch.float)

    data_new = Data(
        x=new_features,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask
    )

    # print(data_new)
    return df_new_features, data_new