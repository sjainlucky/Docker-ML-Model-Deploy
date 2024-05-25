from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
model.load_state_dict(torch.load('app/mnist_cnn.pt', map_location=torch.device('cpu')))
model.eval()

app = FastAPI()

class InputData(BaseModel):
    data: list

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Convert input data to a tensor
        input_tensor = torch.tensor(input_data.data, dtype=torch.float32)

        # Perform the prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            _, predictions = torch.max(prediction, 1)

        return {"prediction": predictions.numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))