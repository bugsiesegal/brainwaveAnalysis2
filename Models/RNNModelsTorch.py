import os
import random
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
import pandas as pd


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], encoding_size: int, dropout=0.0, normalize=False,
                 apply_fft=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.encoding_size = encoding_size
        self.dropout = dropout
        self.normalize = normalize
        self.apply_fft = apply_fft

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Encoder
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.encoder.add_module(f'encoder_{i}', nn.LSTM(input_size, hidden_size, num_layers=1,
                                                               batch_first=True, bidirectional=True))
            else:
                self.encoder.add_module(f'encoder_{i}', nn.LSTM(2 * hidden_sizes[i - 1], hidden_size, num_layers=1,
                                                               batch_first=True, bidirectional=True))

        self.encoder.add_module('encoder_final', nn.LSTM(2 * hidden_sizes[-1], encoding_size,
                                                        num_layers=1, batch_first=True))

        # Decoder
        for i, hidden_size in enumerate(hidden_sizes[::-1]):
            if i == 0:
                self.decoder.add_module(f'decoder_{i}', nn.LSTM(encoding_size, hidden_size, num_layers=1,
                                                               batch_first=True, bidirectional=True))
            else:
                self.decoder.add_module(f'decoder_{i}', nn.LSTM(2 * hidden_sizes[::-1][i - 1], hidden_size, num_layers=1,
                                                               batch_first=True, bidirectional=True))

        self.decoder.add_module('decoder_final',
                                nn.LSTM(2 * hidden_sizes[0], input_size, num_layers=1, batch_first=True))

        # Dense layer
        self.dense1 = nn.Linear(input_size, input_size)
        self.dense2 = nn.Linear(input_size, input_size)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Sigmoid()

    def forward(self, x, return_encoding=False):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply FFT to the input sequence if required
        if self.apply_fft:
            x_fft = torch.fft.fft(x)
            x_real = torch.real(x_fft)
            x_imag = torch.imag(x_fft)
            x_stacked = torch.cat((x_real, x_imag), dim=-1)
        else:
            x_stacked = x

        # Encode input sequence
        for i, encoder_layer in enumerate(self.encoder):
            x_stacked, (hidden_state, cell_state) = encoder_layer(x_stacked)
            if i < len(self.hidden_sizes) - 1:
                x_stacked = x_stacked[:, -1].unsqueeze(1).repeat(1, seq_len, 1)

        x_stacked = self.activation1(x_stacked)

        if return_encoding:
            return x_stacked

        # Decode the encoded sequence
        for i, decoder_layer in enumerate(self.decoder):
            x_stacked, (hidden_state, cell_state) = decoder_layer(x_stacked)
            if i < len(self.hidden_sizes):
                x_stacked = x_stacked[:, -1].unsqueeze(1).repeat(1, seq_len, 1)

        # Apply dense layer
        x_stacked = self.dense1(x_stacked)
        x_stacked = self.dense2(x_stacked)

        # Apply inverse FFT if required
        if self.apply_fft:
            half_dim = x_stacked.size(-1) // 2
            x_real_decoded = x_stacked[..., :half_dim]
            x_imag_decoded = x_stacked[..., half_dim:]
            x_complex_decoded = torch.complex(x_real_decoded, x_imag_decoded)
            x_decoded = torch.fft.ifft(x_complex_decoded).abs()
            return x_decoded
        else:
            x_stacked = self.activation2(x_stacked)
            return x_stacked


def train_model(model, dataset, epochs, batch_size, learning_rate, val_split=0.2, plot_frequency=10,
                save_frequency=10,
                save_path='models', tensorboard_active=False, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0,
                patience=20, factor=0.75, cooldown=0, min_lr=0, verbose=True):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()

    # Move the model to the device
    model.to(device)

    # Split the dataset into training and validation sets
    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),
                           eps=epsilon, amsgrad=False)

    # Set up the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor,
                                                           verbose=verbose, cooldown=cooldown)

    if tensorboard_active:
        log_dir = 'logs'
        writer = SummaryWriter(log_dir=log_dir)

        writer.add_graph(model, input_to_model=train_loader.__iter__().__next__().float().to(device))

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            # Reshape the batch to have the correct dimensions
            batch = batch.float().to(device)

            # Forward pass
            output = model(batch)

            # Calculate the loss
            loss = criterion(output, batch)

            # Backward pass
            loss.backward()

            # Optimize the weights
            optimizer.step()

            # Accumulate the batch loss
            epoch_train_loss += loss.item()

        # Calculate the average loss for this epoch
        epoch_train_loss /= len(train_loader)

        if tensorboard_active:
            for i, encoder_layer in enumerate(model.encoder):
                writer.add_histogram(f'encoder/encoder_{i}/weight_hh_l0', encoder_layer.weight_hh_l0, epoch)
                writer.add_histogram(f'encoder/encoder_{i}/weight_ih_l0', encoder_layer.weight_ih_l0, epoch)
                writer.add_histogram(f'encoder/encoder_{i}/bias_hh_l0', encoder_layer.bias_hh_l0, epoch)
                writer.add_histogram(f'encoder/encoder_{i}/bias_ih_l0', encoder_layer.bias_ih_l0, epoch)

            for i, decoder_layer in enumerate(model.decoder):
                writer.add_histogram(f'decoder/decoder_{i}/weight_hh_l0', decoder_layer.weight_hh_l0, epoch)
                writer.add_histogram(f'decoder/decoder_{i}/weight_ih_l0', decoder_layer.weight_ih_l0, epoch)
                writer.add_histogram(f'decoder/decoder_{i}/bias_hh_l0', decoder_layer.bias_hh_l0, epoch)
                writer.add_histogram(f'decoder/decoder_{i}/bias_ih_l0', decoder_layer.bias_ih_l0, epoch)

            # writer.add_histogram('dense/weight', model.dense1.weight, epoch)
            # writer.add_histogram('dense/bias', model.dense1.bias, epoch)
            #
            # writer.add_histogram('dense2/weight', model.dense2.weight, epoch)
            # writer.add_histogram('dense2/bias', model.dense2.bias, epoch)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Reshape the batch to have the correct dimensions
                batch = batch.float().to(device)

                # Forward pass
                output = model(batch)

                # Calculate the loss
                loss = criterion(output, batch)

                # Accumulate the batch loss
                epoch_val_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        # Log training loss
        wandb.log({"train_loss": epoch_train_loss}, step=epoch + 1, commit=False)

        # Log validation loss
        wandb.log({"val_loss": epoch_val_loss}, step=epoch + 1, commit=False)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, "
            f"Learning rate: {current_lr:.6f}")

        # Update the learning rate scheduler
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % plot_frequency == 0:
            with torch.no_grad():
                # Taking one batch from the validation set to visualize model's performance
                batch = next(iter(val_loader))
                batch = batch.float().to(device)
                output = model(batch)
                encoding = model(batch, return_encoding=True)

                # Choose the first sequence in the batch
                input_sequence = batch[0].squeeze().cpu().numpy()
                output_sequence = output[0].squeeze().cpu().numpy()
                encoding_sequence = encoding[0].squeeze().cpu().numpy()

                plt.plot(input_sequence)
                plt.plot(output_sequence)
                wandb.log({"input/output": plt}, step=epoch + 1)

                plt.close()

                plt.plot(encoding_sequence)
                wandb.log({"encoding": plt}, step=epoch + 1)

                plt.close()

        # Save the model periodically
        if (epoch + 1) % save_frequency == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch + 1}.pth'))
            print(f"Model saved at epoch {epoch + 1}")


def load_model(model_path, input_size, hidden_sizes, encoding_size, dropout=0.0, normalize=False, apply_fft=False):
    # Create a model with the same architecture
    model = RNNAutoencoder(input_size, hidden_sizes, encoding_size, dropout, normalize, apply_fft)

    # Load the state_dict from the saved file
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    return model
