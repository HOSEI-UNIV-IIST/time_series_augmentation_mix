
# Enhanced Self-Attention Transformer (ESAT)

The Enhanced Self-Attention Transformer (ESAT) is a sophisticated neural network model designed for time series classification tasks. It leverages advanced self-attention mechanisms and transformer architectures to effectively capture temporal dependencies and complex patterns within the data. The model comprises several key components:

## Architecture

1. **Learnable Positional Encoding:**
   - Unlike fixed positional encodings, ESAT uses learnable embeddings to encode positional information, allowing the model to adapt to different time series characteristics.

2. **Transformer Encoder:**
   - The encoder consists of multiple layers of Transformer Encoder Layers, each containing multi-head self-attention mechanisms and feedforward neural networks. This structure helps in capturing long-range dependencies and intricate temporal patterns.

3. **Layer Normalization:**
   - Layer normalization is applied after the encoder to stabilize and enhance training performance.

4. **Transformer Decoder:**
   - The decoder mirrors the encoder's structure but focuses on reconstructing the input sequence to predict the final class labels. It ensures that the temporal dependencies captured by the encoder are effectively utilized.

5. **Global Average Pooling:**
   - A global average pooling layer aggregates the output features from the decoder, summarizing the temporal information into a fixed-length representation.

6. **Fully Connected Layer:**
   - The final fully connected layer maps the aggregated features to the target class probabilities, using a softmax activation function for multi-class classification.

## Model Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESAT(nn.Module):
    def __init__(self, input_shape, nb_class, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(ESAT, self).__init__()
        self.input_shape = input_shape
        self.nb_class = nb_class

        # Positional Encoding
        self.pos_encoder = nn.Embedding(input_shape[0], d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, nb_class)

    def forward(self, x):
        # x shape: (batch_size, timesteps, features)
        positions = torch.arange(0, x.size(1)).unsqueeze(0).expand(x.size(0), -1).to(x.device)
        x = x + self.pos_encoder(positions)  # Apply learned positional encoding

        memory = self.transformer_encoder(x)
        memory = self.layer_norm(memory)
        out = self.transformer_decoder(memory, memory)

        out = out.mean(dim=1)  # Global average pooling
        out = self.fc(out)
        return F.softmax(out, dim=1)

# Example usage
input_shape = (128, 64)  # (timesteps, features)
nb_class = 10
model = ESAT(input_shape, nb_class)
```
