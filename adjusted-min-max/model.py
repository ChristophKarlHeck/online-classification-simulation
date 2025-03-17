import pytorch_lightning as pl

# Define the Conv1D Model with LightningModule
class Conv1DModel(pl.LightningModule):
    def __init__(self, input_channels, output_channels, kernel_size, lr):
        super().__init__()
        self.model_name = "CNN"

        self.conv1d_1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.pool = nn.MaxPool1d(3, stride=3) # window size 3, how far windows slided 3

        flattened_size = self._compute_flattened_size(input_channels, output_channels, kernel_size)

        self.linear = nn.Linear(flattened_size, 64, bias=False) # fully connected layer
        self.output = nn.Linear(64, 2, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
        self.lr = lr


    def _compute_flattened_size(self, input_channels, output_channels, kernel_size):
        """Computes the correct flattened size after convolution and pooling layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 100)  # Assume sequence length = 100
            x = self.conv1d_1(dummy_input)
            x = F.relu(x)
            x = self.pool(x)
            x = torch.transpose(x, 1, 2)
            x = torch.cat((x[:, :, 0], x[:, :, 1]), dim=1)  # No flattening, just concatenation
            return x.shape[1]

    def forward(self, x):
        # Ensure input is [batch_size, input_channels, seq_length]
        # Conv Layer
        x = self.conv1d_1(x)
        x = F.relu(x)

        x = self.pool(x) # compress to one convolution block
        # **Flatten**
        x = torch.transpose(x, 1, 2)
        x = torch.cat((x[:,:,0],x[:,:,1]), dim=1)
        #print("Shape before Linear:", x.shape)

        x = self.linear(x)
        x = F.relu(x) # ReLu is not linear. At least one non-linear to recognize non-linear pattern
        x = F.softmax(self.output(x),dim=1) # Sum = 1
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        y = torch.argmax(y, dim=1)
        loss = self.loss_fn(output, y)  # Example loss
        #self.log("train_loss", loss)
        y_pred = torch.argmax(output, dim=1)
        return loss, output, y, y_pred

    def training_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        # print(f"scores: {scores}")
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "train_accuracy": accuracy, 
                "train_scores": scores, "train_y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )        
        return {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def plot(self):
        draw_graph(self, input_size=(1,1,100), expand_nested=True, save_graph=True, filename=self.model_name,
                   directory="results/model_viz/")

    can_delegate = False
