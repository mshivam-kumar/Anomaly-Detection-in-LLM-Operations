import torch.nn as nn

# Step 2: Defining the LSTM Autoencoder Model
class LSTMAutoencoder_without_regularization(nn.Module): # Without regularization
    def __init__(self, input_dim, latent_dim, num_layers=1):
        super(LSTMAutoencoder_without_regularization, self).__init__()
        
        self.input_dim = input_dim   # Number of features
        self.latent_dim = latent_dim # Size of the compressed representation
        self.num_layers = num_layers

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True # This is important!
        )
        
        # --- Decoder ---
        # The decoder LSTM reconstructs the sequence
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=input_dim, # Output should be the original feature dimension
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        
        # --- Encoding ---
        # The encoder outputs the encoded sequence and the final hidden/cell states
        _, (hidden, cell) = self.encoder(x)
        
        # We use the final hidden state as our compressed representation (latent vector)
        # Shape of hidden: (num_layers, batch_size, latent_dim)
        # We want to repeat this vector for each timestep in the sequence for the decoder
        seq_len = x.shape[1]
        
        # Repeat the latent vector `seq_len` times
        # Shape becomes: (batch_size, seq_len, latent_dim)
        latent_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # --- Decoding ---
        reconstructed_seq, _ = self.decoder(latent_repeated)
        
        return reconstructed_seq



# Resolving Overfitting with Dropout
# 1. Introduce Dropout (The Most Effective Regularizer)
# Dropout randomly sets a fraction of neuron activations to zero during training. This forces the network to learn more robust features and prevents neurons from co-adapting too much. It's like forcing a team to work together even if some members are randomly absent, making the whole team stronger.
# We will add dropout to our LSTM layers.
class LSTMAutoencoder_with_dropout_regularization(nn.Module): # With dropout regularization
    def __init__(self, input_dim, latent_dim, num_layers=2, dropout_prob=0.2): # Add dropout_prob
        super(LSTMAutoencoder_with_dropout_regularization, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0 # Dropout is applied between LSTM layers
        )
        
        # --- Decoder ---
        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        seq_len = x.shape[1]
        latent_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed_seq, _ = self.decoder(latent_repeated)
        return reconstructed_seq