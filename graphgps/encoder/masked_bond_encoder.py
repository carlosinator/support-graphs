import torch

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)


@register_edge_encoder('MaskedBond')
class MaskedBondEncoder(torch.nn.Module):
    r"""The bond encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): The output embedding dimension.

    Example:
        >>> encoder = BondEncoder(emb_dim=16)
        >>> batch = torch.randint(0, 10, (10, 3))
        >>> encoder(batch).size()
        torch.Size([10, 16])
    """
    def __init__(self, emb_dim: int, num_hier: int):
        super().__init__()

        from ogb.utils.features import get_bond_feature_dims

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

        self.real_emb = torch.nn.Embedding(num_hier, emb_dim)
        torch.nn.init.xavier_uniform_(self.real_emb.weight.data)

    def forward(self, batch):
        bond_embedding = 0

        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        fake_edge_attr = self.real_emb(batch.real_edge_mask)

        batch.edge_attr = torch.where( # careful not to be in-place
            batch.real_edge_mask == 0, bond_embedding, fake_edge_attr)

        return batch