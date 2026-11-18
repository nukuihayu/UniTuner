"""Compute exact adapter spectra and directions using small QR cores."""

from dataclasses import dataclass

import torch


@dataclass
class FactorMatrix:
    """Keep an exact low-rank matrix factored throughout checkpoint comparisons."""

    left: torch.Tensor
    right: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.right.shape[1]

    def rows(self, start: int, stop: int) -> torch.Tensor:
        return self.left[start:stop] @ self.right

    def subtract(self, other: "FactorMatrix") -> "FactorMatrix":
        return FactorMatrix(torch.cat((self.left, -other.left), dim=1), torch.cat((self.right, other.right)))

    def norm(self) -> float:
        return product_norm(self.left, self.right)

    def projected_energy(self, vectors: torch.Tensor, side: str) -> float:
        if side == "right":
            return product_norm(self.left, self.right @ vectors) ** 2
        return product_norm(vectors.T @ self.left, self.right) ** 2


def product_core(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Reduce L @ R to an orthogonally equivalent core without forming the product."""
    r_left = torch.linalg.qr(left, mode="r").R
    r_right = torch.linalg.qr(right.T, mode="r").R
    return r_left @ r_right.T


def product_norm(left: torch.Tensor, right: torch.Tensor) -> float:
    """Measure Frobenius norm without dense allocation or singular-value computation."""
    return product_core(left, right).norm().item()


def product_basis(left: torch.Tensor, right: torch.Tensor, *, top_k=32, side="right") -> dict:
    """Lift only leading core singular vectors into the requested ambient space.

    Householder QR and LAPACK SVD preserve rank-deficient and rectangular
    products. Only the requested side allocates a reduced Q; the other uses
    R-only QR. No random range finder or output-by-input matrix is needed.
    """
    if side == "left":
        q, r_left = torch.linalg.qr(left, mode="reduced")
        r_right = torch.linalg.qr(right.T, mode="r").R
    else:
        r_left = torch.linalg.qr(left, mode="r").R
        q, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u, values, vh = torch.linalg.svd(core, full_matrices=False)
    tolerance = values[0] * max(left.shape[0], right.shape[1]) * torch.finfo(values.dtype).eps
    k = min(top_k, int((values > tolerance).sum()))
    if not k:
        raise ValueError("Zero matrix has no identifiable singular subspace")
    energy = values.square().sum().item()
    return {
        "vectors": q @ (u[:, :k] if side == "left" else vh[:k].T),
        "values": values[:k],
        "method": "exact QR-core SVD",
        "energy_fraction": min(1.0, values[:k].square().sum().item() / energy),
        "total_energy": energy,
        "rank": k,
    }
