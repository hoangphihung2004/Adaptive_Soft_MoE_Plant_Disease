from torch import nn

class Criterion(nn.Module):

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = self.ce_loss(outputs, targets)
        return loss