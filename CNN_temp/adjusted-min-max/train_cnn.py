import torch
from torchview import draw_graph
from torch import nn
import torch.nn.functional as F
from torch.export import export, export_for_training, ExportedProgram
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
import executorch.exir as exir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import glob
import os
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Conv1DModel(pl.LightningModule):
    def __init__(self, input_channels, output_channels=16, kernel_size=3, hidden_units=32, lr=1e-3):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding="same")
        # self.conv1d_2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding="same")

        #self.dropout_conv = nn.Dropout(0.2)

        self.pool_1 = nn.MaxPool1d(3, stride=3) # Reduce sequence length by half
 
        # Dynamically determine input features to the linear layer
        self.flattened_size = self._compute_flattened_size(input_channels, output_channels, kernel_size)

        self.linear_1 = nn.Linear(self.flattened_size, hidden_units)
        # self.dropout_fc = nn.Dropout(0.4)
        self.output = nn.Linear(hidden_units, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
        

    def _compute_flattened_size(self, input_channels, output_channels, kernel_size):
        """Computes the correct flattened size after convolution and pooling layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 100)  # Assume sequence length = 100
            x = self.conv1d_1(dummy_input)
            x = F.relu(x)
            # x = self.conv1d_2(x)
            # x = F.relu(x)

            #x = self.dropout_conv(x)

            x = self.pool_1(x)
            x = torch.transpose(x, 1, 2)
            x = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)  # No flattening, just concatenation
            return x.shape[1]


    def forward(self, x):
        x = F.relu(self.conv1d_1(x))
        # x = F.relu(self.conv1d_2(x))
        #x = self.dropout_conv(x)
        x = self.pool_1(x)
        x = torch.transpose(x, 1, 2)
        x = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)
        x = F.relu(self.linear_1(x))
        # x = self.dropout_fc(x)
        x = F.softmax(self.output(x), dim=1)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        y = torch.argmax(y, dim=1)
        loss = self.loss_fn(output, y)

        if torch.isnan(loss) or torch.isinf(loss):  # 🚨 Log when loss becomes inf
            print(f"❗ Warning: Loss is {loss.item()} at batch {batch_idx}")

        y_pred = torch.argmax(output, dim=1)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", accuracy, prog_bar=True)
        self.log("test_f1_score", f1_score, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
        return [optimizer], [scheduler]

    @property
    @torch.jit.ignore  # This tells TorchScript to ignore this property
    def trainer(self):
        # Return None or a dummy trainer instance; returning None is usually fine for inference
        return None

# 67,16,5,32,0.00196224387900538,32,9.971408694433246e-05,0.8003876209259033,/home/chris/watchplant_classification_dl/pipline_test/lightning_logs/version_106/checkpoints/epoch=87-step=9152.ckpt
model = Conv1DModel.load_from_checkpoint(
    "/home/chris/watchplant_classification_dl/pipline_test/lightning_logs/version_106/checkpoints/epoch=87-step=9152.ckpt",
    input_channels=1,
    output_channels=16,
    kernel_size=5,
    lr=0.00196224387900538)
    
model = model.cpu()
model.eval()

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') 

# Example Input
final_example_input = (torch.randn(1, 1, 100),)  # [batch_size, input_channels, seq_length]

# Define fixed input tensor should be classified as HEAT=1
fixed_input = torch.tensor([[[ 0.4289, 0.4291, 0.4288, 0.4287, 0.4288, 0.4288, 0.4286, 0.4286, 0.4286,
         0.4286, 0.4288, 0.4288, 0.4288, 0.4287, 0.4284, 0.4282, 0.4280, 0.4279,
         0.4278, 0.4278, 0.4279, 0.4279, 0.4281, 0.4282, 0.4283, 0.4284, 0.4284,
         0.4283, 0.4281, 0.4279, 0.4279, 0.4278, 0.4277, 0.4280, 0.4281, 0.4281,
         0.4281, 0.4281, 0.4284, 0.4284, 0.4285, 0.4286, 0.4286, 0.4287, 0.4288,
         0.4287, 0.4287, 0.4285, 0.4283, 0.4281, 0.4280, 0.4280, 0.4280, 0.4280,
         0.4280, 0.4279, 0.4281, 0.4282, 0.4283, 0.4284, 0.4286, 0.4288, 0.4292,
         0.4295, 0.4295, 0.4296, 0.4296, 0.4295, 0.4293, 0.4291, 0.4290, 0.4290,
         0.4291, 0.4295, 0.4295, 0.4295, 0.4294, 0.4293, 0.4290, 0.4288, 0.4287,
         0.4287, 0.4286, 0.4286, 0.4285, 0.4286, 0.4286, 0.4286, 0.4287, 0.4288,
         0.4289, 0.4291, 0.4293, 0.4294, 0.4295, 0.4295, 0.4294, 0.4294, 0.4294,
         0.4293]]],)

# [[9.9968e-01, 3.1811e-04]]
#output = model(fixed_input)  
# print("Model Input:", fixed_input)
#print("Model Output:", output)

numpy_input = fixed_input[0].detach().cpu().numpy()

# Flatten to a 1D list
flattened_input = numpy_input.flatten().tolist()

# Print the vector representation
print("C++ std::vector<float> representation:")
print(flattened_input)

# Export the Model
pre_autograd_aten_dialect = export_for_training(
        model,
        final_example_input
    ).module()


aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, final_example_input)

print("After:",aten_dialect)

edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)

executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

print("CNN model saved as model.pte")