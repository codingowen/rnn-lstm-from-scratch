import numpy as np

# Read data from .txt file
with open('shakespeare.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
vocab_size = len(chars)
data_size = len(data)
print(f'Data has {data_size} characters in total, with {vocab_size} unique ones')

char_to_ix = {ch : i for i, ch in enumerate(chars)}
ix_to_char = {i : ch for i, ch in enumerate(chars)}

# Set hyperparameters
H = 100 # hidden layer size
T = 25 # Number of time steps to unroll RNN
alpha = 0.1 # Learning rate

# Model parameters
W_hx = np.random.randn(H, vocab_size) * 0.01 # Input x to hidden layer weights
W_hh = np.random.randn(H, H) * 0.01 # Hidden to Hidden recurrent weights
W_yh = np.random.randn(vocab_size, H) * 0.01 # Hidden to output y weights
b_h = np.zeros((H, 1)) # Hidden layer bias
b_y = np.zeros((vocab_size, 1)) # Output layer bias


def loss_function(inputs, targets, h_prev):
    """
    Computes the loss and gradients for backpropagation through time (BPTT).
    """
    x, h, y, p = {}, {}, {}, {} # dict h stores hidden states across time
    h[-1] = np.copy(h_prev)
    loss = 0

    # Forward Pass
    for t in range(len(inputs)):
        x[t] = np.zeros((vocab_size, 1)) # Init zeros for one hot encoding
        x[t][inputs[t]] = 1 # One-hot encoding
        h[t] = np.tanh(np.dot(W_hx, x[t]) + np.dot(W_hh, h[t-1]) + b_h) # Hidden state
        y[t] = np.dot(W_yh, h[t]) + b_y
        p[t] = np.exp(y[t]) / np.sum(np.exp(y[t])) # Softmax probabilities
        loss += -np.log(p[t][targets[t],0]) # Cross entropy loss
    
    # Backward Pass
    dW_hx, dW_hh, dW_yh = np.zeros_like(W_hx), np.zeros_like(W_hh), np.zeros_like(W_yh)
    db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)
    dh_next = np.zeros_like(h[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(p[t])
        dy[targets[t]] -= 1  # Backpropagation through softmax
        dW_yh += np.dot(dy, h[t].T)
        db_y += dy
        dh = np.dot(W_yh.T, dy) + dh_next  # Backpropagation into h
        dz_h = (1 - h[t] ** 2) * dh  # Backpropagation through tanh activation
        db_h += dz_h
        dW_hx += np.dot(dz_h, x[t].T)
        dW_hh += np.dot(dz_h, h[t - 1].T)
        dh_next = np.dot(W_hh.T, dz_h)

    # Gradient clipping to mitigate exploding gradients
    for dparam in [dW_hx, dW_hh, dW_yh, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam) # forces every gradient value to stay in [-5,5]
    
    return loss, dW_hx, dW_hh, dW_yh, db_h, db_y, h[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    Samples a sequence of characters from the model given an initial hidden state.
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    sampled_ix = []
    for t in range(n):
        h = np.tanh(np.dot(W_hx, x) + np.dot(W_hh, h) + b_h)
        y = np.dot(W_yh, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        sampled_ix.append(ix)
    return sampled_ix


# Training loop
n, p = 0, 0
mW_hx, mW_hh, mW_yh = np.zeros_like(W_hx), np.zeros_like(W_hh), np.zeros_like(W_yh)
mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y)  # Adagrad memory
smooth_loss = -np.log(1.0 / vocab_size) * T  # Initial loss

while True:
    if p + T + 1 >= len(data) or n == 0:
        h_prev = np.zeros((H, 1))  # Reset hidden state
        p = 0  # Restart data
    
    inputs = [char_to_ix[ch] for ch in data[p:p+T]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+T+1]]
    
    if n % 100 == 0:
        sample_ix = sample(h_prev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f'----\n Iteration {n}:\n{txt}\n----')
    
    loss, dW_hx, dW_hh, dW_yh, db_h, db_y, h_prev = loss_function(inputs, targets, h_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print(f'Iter {n}, loss: {smooth_loss:.4f}')
    
    for param, dparam, mem in zip([W_hx, W_hh, W_yh, b_h, b_y], 
                                  [dW_hx, dW_hh, dW_yh, db_h, db_y], 
                                  [mW_hx, mW_hh, mW_yh, mb_h, mb_y]):
        mem += dparam * dparam
        param += -alpha * dparam / np.sqrt(mem + 1e-8)  # Adagrad update
    
    p += T
    n += 1