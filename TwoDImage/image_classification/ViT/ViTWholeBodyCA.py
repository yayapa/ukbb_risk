import torch
import torch.nn as nn


# Helper function to create positional encodings
# Helper function to create learnable positional encodings
class LearnablePositionalEncoding3D(nn.Module):
    def __init__(self, num_patches, embed_size):
        super(LearnablePositionalEncoding3D, self).__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(1, num_patches, embed_size))

    def forward(self, x):
        return x + self.positional_encodings


# Cross-Attention mechanism using built-in Transformer layers
class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class ViTWholeBodyCA(nn.Module):
    def __init__(self, in_channels, patch_size, num_classes, embed_size=768, depth=12, num_heads=12,
                 forward_expansion=4, dropout=0.1):
        super(ViTWholeBodyCA, self).__init__()
        self.embed_size = embed_size
        self.patch_size = patch_size

        num_patches = (224 // patch_size[0]) * (168 // patch_size[1]) * (363 // patch_size[2])
        self.water_pos_embedding = LearnablePositionalEncoding3D(num_patches, embed_size)
        self.fat_pos_embedding = LearnablePositionalEncoding3D(num_patches, embed_size)

        self.water_embedding = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_size)
        self.fat_embedding = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_size)

        self.transformer_water = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, forward_expansion * embed_size, dropout,
                                       batch_first=True), depth
        )
        self.transformer_fat = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, forward_expansion * embed_size, dropout,
                                       batch_first=True), depth
        )
        self.cross_attention = CrossAttention(embed_size, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    def forward(self, x):
        #print("Dimensions of input tensor:", x.shape)
        water, fat = x[:, 0, :, :, :], x[:, 1, :, :, :]

        # Patch extraction
        water_patches = water.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1]).unfold(3, self.patch_size[2], self.patch_size[2])
        fat_patches = fat.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1]).unfold(3, self.patch_size[2], self.patch_size[2])

        water_patches = water_patches.contiguous().view(water_patches.size(0), -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        fat_patches = fat_patches.contiguous().view(fat_patches.size(0), -1, self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

        # Embedding
        water_embedded = self.water_embedding(water_patches)
        fat_embedded = self.fat_embedding(fat_patches)

        # Add positional encoding
        water_embedded = self.water_pos_embedding(water_embedded)
        fat_embedded = self.fat_pos_embedding(fat_embedded)

        # Transformer Encoding
        water_encoded = self.transformer_water(water_embedded)
        fat_encoded = self.transformer_fat(fat_embedded)

        # Cross-Attention
        water_cross = self.cross_attention(water_encoded, fat_encoded, fat_encoded)
        fat_cross = self.cross_attention(fat_encoded, water_encoded, water_encoded)

        # Combine features and apply final transformer
        combined_features = torch.cat((water_cross, fat_cross), dim=1)
        combined_features = combined_features.mean(dim=1)  # Global Average Pooling

        # Classification
        out = self.fc(combined_features)
        out.squeeze(1)
        return out

# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTWholeBodyCA(in_channels=1, patch_size=(16, 16, 16), num_classes=1, device=device).to(device)

    # Dummy input
    x = torch.randn((6, 2, 224, 168, 369)).to(device)  # Batch size of 2
    preds = model(x)
    print(preds.shape)  # Expected output: (2, 1)
