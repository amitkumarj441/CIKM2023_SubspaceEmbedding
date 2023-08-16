import torch
from torch import device, nn, Tensor, 

class SubspaceEmbedding(nn.Module):

    def __init__(self, n_embeddings, embedding_dim, n_spaces=2, **kwargs):
        """
        f-subspace embedding layer

        :param n_embeddings: the number of entire embeddings
        :param embedding_dim: dimension of the embedding (not subspace embedding)
        :param n_spaces: the number of partitioned subspace embedding.
        :param kwargs: parameters follow as per nn.Embedding
        """

        super(SubspaceEmbedding, self).__init__()

        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.n_spaces = n_spaces
        # The number of each subspace embedding
        n = int(math.pow(n_embeddings, 1 / n_spaces)) + 1 # n = exp(log(N) / f) + 1
        self.n = n

        n_embedding_list = [n] * self.n_spaces
        s_embedding_dims = [embedding_dim // n_spaces] * self.n_spaces
        s_embedding_dims[-1] = embedding_dim - (embedding_dim//n_spaces) * (n_spaces-1)

        # If embedding layer set to have padding index, then set dummy weight to embedding,
        # and allocate the last subspace embedding to padding index.
        embeddings = nn.ModuleList()
        padding_idx = kwargs.get("padding_idx", 1)
        for i in range(n_spaces):
            if "padding_idx" in kwargs:
                kwargs.update({"padding_idx": n_embedding_list[i]})
                embeddings.append(nn.Embedding(n_embedding_list[i] + 1, s_embedding_dims[i], **kwargs))
            else:
                embeddings.append(nn.Embedding(n_embedding_list[i], s_embedding_dims[i], **kwargs))

        self.embeddings = embeddings
        # all subspace embedding layers are initialised independently
        for embedding in self.embeddings:
            embedding.weight.data.uniform_(-0.1, 0.1)

        mapper = self.set_mapper()
        if "padding_idx" in kwargs:
            for i in range(n_spaces):
                mapper[padding_idx, i] = n_embedding_list[i]
        self.register_buffer("mapper", mapper)

    def set_mapper(self) -> Tensor:
        mapper = torch.zeros(self.n_embeddings, self.n_spaces).long()
        idx = torch.arange(self.n_embeddings)

        for space in range(self.n_spaces):
            mapper[:, space] = torch.remainder(idx, self.n).long()
            idx = torch.div(idx, self.n)
        return mapper

    def forward(self, ipt: Tensor) -> Tensor:
        embedding_idx = []

        # lookup each subspace embedding and concatenate them
        for space in range(self.n_spaces):
            space_idx = self.mapper[:, space]
            embedding_idx.append(torch.nn.functional.embedding(ipt, space_idx))
        output = torch.cat([emb(idx) for emb, idx in zip(self.embeddings, embedding_idx)], dim=-1)
        return output


class TiledLinear(nn.Linear):

    def __init__(self, input_embeddings, *args, **kwargs):
        """
        This layer is used for tie-weight in subspace embedding
        :param input_embeddings: nn.Embedding (Subspace Embedding layer)
        """
        in_features = input_embeddings.embedding_dim
        out_features = input_embeddings.n_embeddings
        super(TiledLinear, self).__init__(in_features, out_features, *args, **kwargs)
        self.input_embeddings = input_embeddings
        if isinstance(input_embeddings, nn.Embedding):
            self.weight = input_embeddings.weight
        else:
            self.tile_weight()

    def tile_weight(self):
        assert isinstance(self.input_embeddings, SubspaceEmbedding)
        n = self.input_embeddings.n
        idx = torch.arange(self.input_embeddings.n_embeddings)

        embedding_idx = []
        for space in range(self.input_embeddings.n_spaces):
            embedding_idx.append(torch.remainder(idx, n).long())
            idx = torch.div(idx, n)

        embeddings = self.input_embeddings.embeddings
        weight = torch.cat([emb.weight[idx] for emb, idx in zip(embeddings, embedding_idx)], dim=-1)
        return weight

    def forward(self, ipt: Tensor) -> Tensor:
        # update the weight each train step
        if isinstance(self.input_embeddings, SubspaceEmbedding):
            weight = self.tile_weight()
        else:
            weight = self.weight
        return torch.nn.functional.linear(ipt, weight, self.bias)
