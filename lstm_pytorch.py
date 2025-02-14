import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data I/O
data = open('shakespeare.txt', 'r').read()
chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
hidden_size = 512  # Hidden state size
num_layers = 3  # Two-layer LSTM
seq_length = 60  # Sequence length
learning_rate = 1e-3  # Learning rate
batch_size = 64  # Batch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embed(x)  # Convert to embeddings
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

# Initialize model, loss, and optimizer
model = CharLSTM(vocab_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 5000  # Number of iterations
hprev = model.init_hidden(batch_size)

for epoch in range(num_epochs):
    if (epoch * seq_length + seq_length >= len(data)):
        hprev = model.init_hidden(batch_size)  # Reset hidden state
        start = 0
    else:
        start = epoch * seq_length
    
    inputs = []
    targets = []

    for i in range(batch_size):
        start_idx = (epoch * seq_length + i * seq_length) % (len(data) - seq_length)
        input_seq = [char_to_ix[ch] for ch in data[start_idx:start_idx+seq_length]]
        target_seq = [char_to_ix[ch] for ch in data[start_idx+1:start_idx+seq_length+1]]
        
        inputs.append(input_seq)
        targets.append(target_seq)

    inputs = torch.tensor(inputs, dtype=torch.long).to(device)  # Shape: (batch_size, seq_length)
    targets = torch.tensor(targets, dtype=torch.long).to(device)  # Shape: (batch_size, seq_length)

    
    model.zero_grad()
    hprev = tuple([h.detach() for h in hprev])
    outputs, hprev = model(inputs, hprev)
    loss = criterion(outputs, targets.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Sample text
        sample_input = torch.tensor([char_to_ix[data[start]]], dtype=torch.long).unsqueeze(0).to(device)
        h_sample = model.init_hidden(1)
        sampled_chars = []
        
        for _ in range(200):  # Generate 200 characters
            output, h_sample = model(sample_input, h_sample)
            prob = torch.nn.functional.softmax(output[-1], dim=0).detach().cpu().numpy()
            char_index = np.random.choice(vocab_size, p=prob)
            sampled_chars.append(ix_to_char[char_index])
            sample_input = torch.tensor([[char_index]], dtype=torch.long).to(device)
        
        print("----\n" + ''.join(sampled_chars) + "\n----")