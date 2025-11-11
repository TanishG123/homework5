import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        # code help from ChatGPT
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.d = int(d_latent)

        # token embed + learned start vector
        self.token_embed = torch.nn.Embedding(self.n_tokens, self.d)
        self.start_embed = torch.nn.Parameter(torch.zeros(1, 1, self.d))

        # simple learned positional embedding (big enough for 30*20=600)
        self.max_len = 1200
        self.pos_embed = torch.nn.Embedding(self.max_len, self.d)

        # a small decoder-only stack using TransformerEncoder layers with a causal mask
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=8,
            dim_feedforward=4 * self.d,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.backbone = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        # projection to vocabulary
        self.head = torch.nn.Linear(self.d, self.n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # code help from ChatGPT
        assert x.dtype in (torch.long, torch.int64), "Input tokens must be integer (long)."
        B, h, w = x.shape
        L = h * w

        # flatten to sequence
        x_seq = x.view(B, L)                                 # (B, L)

        # token embedding
        tok = self.token_embed(x_seq)                        # (B, L, d)

        # shift right by 1 with a learned start vector at position 0
        start = self.start_embed.expand(B, 1, self.d)        # (B, 1, d)
        tok_shifted = torch.cat([start, tok[:, :-1, :]], dim=1)  # (B, L, d)

        # (optional) positional embedding
        pos_idx = torch.arange(L, device=x.device)
        pos = self.pos_embed(pos_idx)[None, :, :]            # (1, L, d)
        src = tok_shifted + pos                              # (B, L, d)

        # causal mask (allow attending to <= current position)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L).to(x.device)  # (L, L)

        # transformer encoder acts as a decoder when given a causal mask + shifted inputs
        hseq = self.backbone(src, mask=mask)                 # (B, L, d)

        # project to logits
        logits = self.head(hseq)                             # (B, L, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # code help from ChatGPT
        if device is None:
            device = next(self.parameters()).device

        L = h * w
        # initialize sequence with zeros (any valid token id is fine)
        seq = torch.zeros(B, L, dtype=torch.long, device=device)

        # generate left-to-right
        for t in range(L):
            # forward expects (B,h,w) -> supply the current seq as a grid
            logits, _ = self.forward(seq.view(B, h, w))
            # grab logits for position t
            step_logits = logits.view(B, L, self.n_tokens)[:, t, :]  # (B, n_tokens)
            # greedy pick
            next_tok = step_logits.argmax(dim=-1)                    # (B,)
            seq[:, t] = next_tok

        return seq.view(B, h, w)
