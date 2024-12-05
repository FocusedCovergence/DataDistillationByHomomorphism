import torch

def add_mask(data): 
    if hasattr(data, 'train_mask') and (data.train_mask is not None) and False:
        # Dataset already has train/validation/test masks
        ################################################
        ## Forbid this part for forcing add new masks ##
        ################################################
        print("Dataset already has train, validation, and test masks.")
    else:
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else len(data)
        
        # reset mask
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        num_train = int(0.8 * num_nodes)
        num_val = int(0.1 * num_nodes)
        num_test = int(0.1 * num_nodes)
        
        # shuffle indices
        indices = torch.randperm(num_nodes)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:num_train + num_val + num_test]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print("Added train, validation, and test masks with 8:1:1 split.")
    return(data)