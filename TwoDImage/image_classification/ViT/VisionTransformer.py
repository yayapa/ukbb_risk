import torch.nn as nn
import torch
import numpy as np


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=128,
        slice_num=8,
        time_frame=50,
        patch_size=(16, 16),
        embed_dim=1024,
        depth=12,
        num_heads=8,
        in_channels=1,
        num_classes=10,
        norm_layer=nn.LayerNorm,
        use_both_axes=False,
    ):
        super(VisionTransformer, self).__init__()

        self.img_3dt_shape = (slice_num, time_frame, img_size, img_size)
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            self.img_3dt_shape, in_channels, patch_size, embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim),
            requires_grad=False,
        )

        # Transformer Encoder
        self.encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    activation="gelu",
                )
                for _ in range(depth)
            ]
        )
        self.encoder_norm = norm_layer(embed_dim)

        # Classification Head
        if num_classes == 2:
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        initialize_weights(self, use_both_axes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : N + 1, :]
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)
        cls_token_final = x[:, 0]
        out = self.head(cls_token_final)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_multi_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_dim = len(grid_size)
    #print("grid_size: ", grid_size)
    #print("embed_dim: ", embed_dim)
    #print("grid_dim: ", grid_dim)
    assert grid_dim >= 2, "Grid_size should be at least 2D"
    assert embed_dim % (grid_dim * 2) == 0, "Each dimension has 2 channels (sin, cos)"

    grid = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in grid_size], indexing="ij"
    )
    grid = torch.stack(grid, dim=0)
    pos_embed = get_multi_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_multi_sincos_pos_embed_from_grid(embed_dim, grid):
    grid_dim = len(grid.shape) - 1
    emb = [
        get_1d_sincos_pos_embed_from_grid(embed_dim // grid_dim, grid[i])
        for i in range(grid.shape[0])
    ]
    emb = torch.cat(emb, dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


def get_multi_sincos_pos_embed_with_both_axes(embed_dim, grid_size, cls_token=False):
    """
    Generate multi-scale sine-cosine position embedding for 3D+t images with both long axis and short axis slices.

    grid_size: tuple of grid dimensions (..., H, W)
    embed_dim: output dimension for each position
    pos_embed: [np.prod(grid_size), embed_dim] or [1+np.prod(grid_size), embed_dim] (w/ or w/o cls_token)
    """
    grid_dim = len(grid_size)
    grid_size_sax = (grid_size[0] - 3, *grid_size[1:])
    grid_size_lax = (3, *grid_size[1:])
    #print("Grid size short axis: ", grid_size_sax)
    #print("Grid size long axis: ", grid_size_lax)
    #print("grid_dim: ", grid_dim)
    #print("grid_dim: ", grid_dim)
    assert (
        grid_dim >= 3
    ), "Grid_size should be at least 3D for positional embedding with long axis"
    assert (embed_dim - 1) % (
        grid_dim * 2
    ) == 0, "Each dimension has 2 channels (sin, cos)"

    # Get long axis position embedding
    grid_lax = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in grid_size_lax], indexing="ij"
    )
    grid_lax = torch.stack(grid_lax, dim=0)
    pos_embed_lax = get_multi_sincos_pos_embed_from_grid(embed_dim - 1, grid_lax)

    # Get short axis position embedding
    grid_sax = torch.meshgrid(
        *[torch.arange(s, dtype=torch.float32) for s in grid_size_sax], indexing="ij"
    )
    grid_sax = torch.stack(grid_sax, dim=0)
    pos_embed_sax = get_multi_sincos_pos_embed_from_grid(embed_dim - 1, grid_sax)

    # Concatenate long axis and short axis position embedding and add axis distinguishing embedding
    ax_pos = torch.cat(
        [
            torch.zeros([pos_embed_lax.shape[0], 1]),
            torch.ones([pos_embed_sax.shape[0], 1]),
        ],
        dim=0,
    )
    pos_embed = torch.cat([pos_embed_lax, pos_embed_sax], dim=0)
    pos_embed = torch.cat([ax_pos, pos_embed], dim=1)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


class PatchEmbed(nn.Module):
    def __init__(
        self,
        im_shape,
        in_channels=1,
        patch_size=16,
        embed_dim=256,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.im_shape = im_shape
        self.patch_size = patch_size
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        assert self.in_channels == 1, "Patch size should be 2D for temporal embedding"
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.grid_size = (
            im_shape[0],
            im_shape[1] // patch_size[0],
            im_shape[2] // patch_size[1],
            im_shape[3] // patch_size[2],
        )

        self.num_patches = np.prod(self.grid_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, S, T = x.shape[:3]
        if self.in_channels == 1:
            x = x.reshape(-1, *self.im_shape[-len(self.patch_size) :])
            assert x.shape[-2:] == self.im_shape[-2:]
            x = x.unsqueeze(1)
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(len(self.patch_size)).transpose(1, 2)
            x = x.reshape(B, -1, self.embed_dim)
        x = self.norm(x)
        return x


def initialize_weights(model, use_both_axes):
    pos_embed_fctn = (
        get_multi_sincos_pos_embed
        if not use_both_axes
        else get_multi_sincos_pos_embed_with_both_axes
    )
    pos_embed = pos_embed_fctn(
        model.embed_dim, model.patch_embed.grid_size, cls_token=True
    )
    model.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    w = model.patch_embed.proj.weight.data
    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    if hasattr(model, "cls_token"):
        torch.nn.init.normal_(model.cls_token, std=0.02)
    if hasattr(model, "mask_token"):
        torch.nn.init.normal_(model.mask_token, std=0.02)

    model.apply(model._init_weights)
