import math
import torch
import torch.nn as nn
from torch.nn import functional as F

"""
World GPT: A mathematically intuitive implementation of a Generative Pre-trained Transformer.

The objective of this file is to translate the core equations of a Transformer into code, 
while explaining *why* the math is structured this way. Instead of viewing the transformer 
as a black box of layers, we will look at it as a sequence of geometric transformations 
that route information through a graph.
"""

class CausalSelfAttention(nn.Module):
    """
    Self-Attention is the heart of the Transformer. It is basically a communication phase.
    
    Imagine a room full of words. Every word wants to update its meaning based on the 
    context of other words around it. But words can only look at words that came *before* them 
    (because it is causal/autoregressive).
    
    Mathematically, we route information using three vectors for each token:
    1. Query (Q): What I am looking for (e.g., "I am an adjective looking for a noun")
    2. Key (K): What I contain (e.g., "I am a noun, and I describe a dog")
    3. Value (V): What I will actually communicate to you if we match.
    """
    def __init__(self, d_model, n_head, max_seq_len):
        super().__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # We project the input into Q, K, V simultaneously for efficiency.
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # The causal mask ensures token i can only attend to tokens j where j <= i.
        # We use a lower triangular matrix for this. In log-space, adding -inf means 
        # multiplying by 0 after the exponentiation in softmax.
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length (Time), Embedding Dimension (Channels)

        # 1. Linear projection and head splitting
        # x is (B, T, C). We project to 3*C, then split into Q, K, V.
        # Each is then reshaped to (B, n_head, T, d_head) so heads process in parallel.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2) 

        # 2. Compute the Attention Scores (The Math)
        # We want to know how much query i matches key j. We do this via dot product: q_i \cdot k_j
        # In matrix form: Q @ K^T. 
        # A dot product measures cosine similarity (scaled by magnitudes). If Q and K point 
        # in the same direction in d_head-dimensional space, the score is high.
        
        # Why divide by sqrt(d_head)? 
        # If Q and K have unit variance, their dot product has a variance of d_head. 
        # If d_head is large, the dot product values become huge. E.g., variance of 64 means std dev of 8.
        # Feeding large numbers into Softmax pushes the outputs to 1 and 0 (a one-hot vector) 
        # which means the gradients vanish (flat slopes). Dividing by sqrt(d_head) 
        # mathematically preserves a variance of 1, keeping Softmax in its soft, sensitive region.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        
        # 3. Apply the Causal Mask
        # We replace the upper triangle of the attention matrix with negative infinity.
        # Softmax applies e^x. Since e^{-infinity} = 0, future tokens get exactly 0 weight, 
        # so no information can flow backwards in time.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # 4. Softmax
        # Softmax turns raw continuous scores into a discrete probability distribution.
        # For a given row (a given query), the sum of attention to all past keys equals 1.0.
        att = F.softmax(att, dim=-1)

        # 5. Aggregate the Values
        # Now that we know who to pay attention to (the probabilities in `att`), 
        # we take a weighted sum of the Values. 
        # Mathematically: Out = att @ V
        y = att @ v # (B, n_head, T, T) @ (B, n_head, T, d_head) -> (B, n_head, T, d_head)
        
        # Re-assemble the heads back into a single vector of size C
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear projection to mix the outputs of the different heads
        y = self.c_proj(y)
        return y


class FeedForward(nn.Module):
    """
    If Attention is the "communication" phase between tokens, the FeedForward network 
    is the "computation" or "reflection" phase for each token individually.
    
    Mathematically, it is an MLP (Multi-Layer Perceptron) applied to every token independently.
    It expands the dimensionality (typically by a factor of 4) to give the model lots of 
    parameters to store learned patterns and facts, then projects back down.
    """
    def __init__(self, d_model):
        super().__init__()
        self.c_fc   = nn.Linear(d_model, 4 * d_model)
        # Gaussian Error Linear Unit (GELU)
        # Unlike ReLU which is a strict max(0, x), GELU is mathematically smoothed.
        # It weights x by the cumulative distribution function of the Gaussian distribution: x * Phi(x)
        # This provides a smoother gradient landscape than ReLU's sharp corner at 0.
        self.act    = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    """
    A Transformer block chains together Attention (communication) and FeedForward (computation).
    """
    def __init__(self, d_model, n_head, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        # RESIDUAL CONNECTIONS (the + operator)
        # Mathematically, we compute: x = x + Layer(x)
        # 
        # Why? Imagine a deep network. By the chain rule, a gradient flowing backwards 
        # is a series of multiplications. Multiplying many small numbers leads to vanishing gradients.
        # The + operator in calculus acts as an "addition router" for gradients:
        # d(x + f(x))/dx = 1 + f'(x)
        # This explicit "1" means the gradient flows backwards through the network uncorrupted, 
        # acting like a superhighway for gradients directly from the loss function to the earliest layers.
        
        # PRE-NORMALIZATION (LayerNorm before the layer)
        # Layer Normalization stabilizes the hidden state dynamics. 
        # Math: We calculate the mean and variance across the embedding dimension (C).
        # We subtract the mean and divide by standard deviation: (x - u) / (std).
        # This forces the features to have roughly zero mean and unit variance.
        # It ensures that massive activations in one channel don't completely drown out the others,
        # which keeps the gradient flow remarkably stable.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class WorldGPT(nn.Module):
    """
    The full World GPT model.
    It links token IDs to continuous vectors, passes them through physical blocks of geometry 
    (the blocks), and maps them back into a vocabulary probability space.
    """
    def __init__(self, vocab_size, d_model=768, n_layer=12, n_head=12, max_seq_len=1024):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # 1. Embedding Layer
        # Discrete symbols (like word ID 452) cannot be differentiated. 
        # We look up a learnable continuous vector for every discrete word token.
        self.wte = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        # Attention is functionally a set-operation, meaning it has no concept of order. 
        # Q @ K^T yields the same result no matter where the tokens sit in the sequence.
        # Thus, we give each physical position its own learnable vector, and ADd it 
        # to the token's vector. Now, the vector contains both "What" it is and "Where" it is.
        self.wpe = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.Sequential(*[Block(d_model, n_head, max_seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        
        # 3. The Language Model Head
        # Projects the final d_model dimensional vector back to vocab_size dimensions
        # to predict the next word.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: 
        # Conceptually, the transformation to project a token into a vector space (self.wte)
        # is the exact inverse geometry needed to take a meaning vector and pick a word (self.lm_head).
        # Tying them saves enormous amounts of memory and improves learning.
        self.wte.weight = self.lm_head.weight 

        # Initialize weights mathematically
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Normal distribution with standard deviation 0.02
            # 0.02 comes from ~ 1/sqrt(d_model) for wide networks, ensuring unit variance 
            # flows through the network initially.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        
        # Add discrete structural vectors (What + Where)
        tok_emb = self.wte(idx) # (B, T, C)
        pos_emb = self.wpe(pos) # (T, C)
        x = tok_emb + pos_emb   # Broadcasting applies pos_emb to every item in the Batch
        
        # Flow through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get raw, un-normalized predictions (logits)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # CROSS ENTROPY LOSS / NEGATIVE LOG LIKELIHOOD 
            # Let's say the correct target word has an index of `k`.
            # We want the probability of `k` to be exactly 1.0, and 0 for everything else.
            # 
            # How do we measure distance between our predicted probability distribution and reality?
            # We use Cross Entropy.
            # 
            # Math: Loss = -log(P(target))
            # If the model assigns P(target) = 1.0, then -log(1.0) = 0. (Perfect score).
            # If the model assigns P(target) = 0.001, then -log(0.001) is a large positive number.
            # The gradient naturally tells the weights how to shift the geometric space so that
            # the logits for the correct target index are higher than the rest.
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generation loop. Autoregressive inference.
        Take sequence -> output prediction -> append to sequence -> repeat.
        """
        for _ in range(max_new_tokens):
            # Condense context if it gets too long
            idx_cond = idx[:, -self.max_seq_len:]
            # Provide current sequence to get predictions
            logits, _ = self(idx_cond)
            # We only care about the very last token's prediction in the Time dimension
            logits = logits[:, -1, :] # (B, vocab_size)
            # Exponentiate and normalize logits to get legitimate probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append generated token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

