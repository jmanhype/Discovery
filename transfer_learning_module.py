import torch
import torch.nn.functional as F
from typing import Tuple


class TransferLearningModule:
    def __init__(self, source_model, target_model, source_optimizer, device='cpu'):
        self.source_model = source_model
        self.target_model = target_model
        self.source_optimizer = source_optimizer
        self.device = device

    def _weighted_loss(self, source_loss: torch.Tensor, target_loss: torch.Tensor, lambda_weight: float) -> torch.Tensor:
        return source_loss * (1 - lambda_weight) + target_loss * lambda_weight

    def train_source_model(self, source_data_loader, epochs: int):
        self.source_model.train()

        for epoch in range(epochs):
            for batch_idx, (data, labels) in enumerate(source_data_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.source_optimizer.zero_grad()
                outputs = self.source_model(data)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.source_optimizer.step()

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Source model training batch {batch_idx}/{len(source_data_loader)}, Loss: {loss.item()}")

    def transfer_learning(self, target_data_loader, source_to_target_mapping: dict, lambda_weight: float, epochs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.target_model.train()

        target_losses = []
        source_losses = []

        for epoch in range(epochs):
            for batch_idx, (data, labels) in enumerate(target_data_loader):
                data, labels = data.to(self.device), labels.to(self.device)

                self.source_optimizer.zero_grad()
                target_outputs = self.target_model(data)

                target_loss = F.cross_entropy(target_outputs, labels)
                source_loss = 0
                for target_label, source_label in source_to_target_mapping.items():
                    source_loss += F.mse_loss(self.source_model(data).detach()[:, source_label],
                                              self.target_model(data).detach()[:, target_label])

                weighted_loss = self._weighted_loss(source_loss, target_loss, lambda_weight)
                weighted_loss.backward()
                self.source_optimizer.step()

                if batch_idx % 10 == 0:
                    target_losses.append(target_loss.item())
                    source_losses.append(source_loss.item())
                    print(f"Epoch {epoch}, Transfer learning batch {batch_idx}/{len(target_data_loader)}, Target loss: {target_loss.item()}, Source loss: {source_loss.item()}")

        return torch.Tensor(target_losses), torch.Tensor(source_losses)