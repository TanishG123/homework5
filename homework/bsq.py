import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae import PatchAutoEncoder, hwc_to_chw, chw_to_hwc


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        # code help from ChatGPT
        super().__init__()
        self._codebook_bits = int(codebook_bits)
        self._embedding_dim = int(embedding_dim)
        self.down = torch.nn.Linear(self._embedding_dim, self._codebook_bits, bias=False)
        self.up = torch.nn.Linear(self._codebook_bits, self._embedding_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # code help from ChatGPT
        s = x.shape
        y = self.down(x.reshape(-1, s[-1])).reshape(*s[:-1], self._codebook_bits)
        # L2-normalize before binarization
        y_norm = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(1e-6)
        y = y / y_norm
        code = diff_sign(y)
        return code

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        # code help from ChatGPT
        s = x.shape
        y = self.up(x.reshape(-1, s[-1])).reshape(*s[:-1], self._embedding_dim)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        # code help from ChatGPT
        # Keep AE bottleneck at latent_dim; BSQ lives between encoder and decoder
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = int(codebook_bits)
        self.bsq = BSQ(codebook_bits=self.codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        # code help from ChatGPT
        # x: (B,H,W,3) in [-0.5,0.5] -> (B,h,w) ints
        z = self.encoder(x)                 # (B,h,w,latent_dim)
        code = self.bsq.encode(z)           # (B,h,w,Cbits) in {-1,+1}
        idx = self.bsq._code_to_index(code) # (B,h,w)
        return idx

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        # code help from ChatGPT
        # x: (B,h,w) ints -> (B,H,W,3)
        code = self.bsq._index_to_code(x)   # (B,h,w,Cbits) in {-1,+1}
        z = self.bsq.decode(code)           # (B,h,w,latent_dim)
        y = self.decoder(z)                 # (B,H,W,3)
        return y

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # code help from ChatGPT
        # Return the binary code (B,h,w,Cbits) for inspection
        z = self.encoder(x)
        code = self.bsq.encode(z)
        return code

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # code help from ChatGPT
        # If last dim is code bits, first BSQ-decode to features; else assume features
        if x.size(-1) == self.codebook_bits:
            z = self.bsq.decode(x)          # (B,h,w,latent_dim)
        else:
            z = x                           # already features
        y = self.decoder(z)                 # (B,H,W,3)
        return y

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # code help from ChatGPT
        # AE encode -> BSQ binarize -> BSQ decode -> AE decode
        z = self.encoder(x)                 # (B,h,w,latent_dim)
        code = self.bsq.encode(z)           # (B,h,w,Cbits)
        zq = self.bsq.decode(code)          # (B,h,w,latent_dim)
        x_hat = self.decoder(zq)            # (B,H,W,3)

        # Optional diagnostics for TensorBoard
        with torch.no_grad():
            idx = self.bsq._code_to_index(code)
            K = 2 ** self.codebook_bits
            cnt = torch.bincount(idx.flatten(), minlength=K).float()
            logs = {
                "cb_unused_frac": (cnt == 0).float().mean(),
                "cb_rare_frac":   (cnt <= 2).float().mean(),
                "code_abs_mean":  code.abs().mean(),
            }
        return x_hat, logs
