import torch
from torch import nn

from typing import Sequence, List, Tuple
from torch import Tensor

from .builder import HEAD


@HEAD.register()
class TokenHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_interactions: int,
        token_feature_dim: int,
        num_feature_mlp_layers: int,
        num_score_mlp_layers: int,
    ):
        super(TokenHead, self).__init__()
        self.interaction_embedding = nn.Embedding(num_interactions, feature_dim)
        self.token_feature_dim = token_feature_dim

        feature_mlp = []
        dim = 2 * feature_dim
        for _ in range(num_feature_mlp_layers):
            feature_mlp.append(nn.Linear(dim, token_feature_dim))
            feature_mlp.append(nn.SiLU(inplace=True))
            dim = token_feature_dim
        self.feature_mlp = nn.Sequential(*feature_mlp)
        if 2 * feature_dim != token_feature_dim:
            self.skip = nn.Linear(2 * feature_dim, token_feature_dim)
        else:
            self.skip = nn.Identity()

        score_mlp = []
        for _ in range(num_score_mlp_layers - 1):
            score_mlp.append(nn.Linear(token_feature_dim, token_feature_dim))
            score_mlp.append(nn.ReLU(inplace=True))
        score_mlp.append(nn.Linear(token_feature_dim, 1))
        self.score_mlp = nn.Sequential(*score_mlp)

    def initialize_weights(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.uniform_(self.interaction_embedding.weight, -1.0, 1.0)
        for m in [self.feature_mlp, self.score_mlp, self.skip]:
            m.apply(_init_weight)

    def forward(self, features: Tensor, tokens_list: Sequence[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Token Scoring Function

        Args:
            features: FloatTensor [N, F, D, H, W]
            tokens_list: List[IntTensor [Ntoken, 4] - (x, y, z, i)]

        Returns:
            token_scores_list: List[FloatTensor [Ntoken,]]
            token_features_list: List[FloatTensor [Ntoken, F]]
        """
        num_images = len(tokens_list)
        token_features_list = [self.extract_token_features(features[idx], tokens_list[idx]) for idx in range(num_images)]
        token_scores_list = [self.score_mlp(token_features).squeeze(-1) for token_features in token_features_list]
        return token_scores_list, token_features_list

    def extract_token_features(self, features: Tensor, tokens: Tensor) -> Tensor:
        """Extract token features

        Args:
            features: FloatTensor [D, H, W, F]
            tokens: IntTensor [Ntoken, 4] - (x, y, z, i)

        Returns:
            token_features: FloatTensor [Ntoken, Fh]
        """
        if tokens.size(0) == 0:
            return torch.empty([0, self.token_feature_dim], dtype=torch.float, device=features.device)
        else:
            features = features.permute(1, 2, 3, 0).contiguous()                    # [D, H, W, F]
            x_list, y_list, z_list, i_list = torch.split(tokens, 1, dim=1)          # (x_list, y_list, z_list, i_list)
            token_features = features[x_list, y_list, z_list].squeeze(1)            # [Ntoken, F]
            embeddings = self.interaction_embedding(i_list).squeeze(1)              # [Ntoken, F]
            token_features = torch.cat([token_features, embeddings], dim=1)         # [Ntoken, 2F]
            return self.skip(token_features) + self.feature_mlp(token_features)     # [Ntoken, Fh]
