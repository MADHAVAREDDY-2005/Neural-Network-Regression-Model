# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: K MADHAVA REDDY
### Register Number: 212223240064
```python
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=nn.Linear(1,6)
    self.l2=nn.Linear(6,12)
    self.l3=nn.Linear(12,20)
    self.l4=nn.Linear(20,1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}
  def forward(self,x):
    x=self.relu(self.l1(x))
    x=self.relu(self.l2(x))
    x=self.relu(self.l3(x))
    x=self.l4(x)
    return x


my_model=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(my_model.parameters(),lr=0.001)

def train_model(my_model, X_train, y_train, criterion, optimizer, epochs=2000):
  for i in range(epochs):
    optimizer.zero_grad()
    loss=criterion(my_model(X_train),y_train)
    loss.backward()
    optimizer.step()
    my_model.history['loss'].append(loss.item())
    if i % 200 == 0:
      print(f'Epoch [{i}/{epochs}], Loss: {loss.item():.6f}')

train_model(my_model, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(my_model(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(my_model.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[20]], dtype=torch.float32)
prediction = my_model(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
## Dataset Information
<img width="722" height="867" alt="image" src="https://github.com/user-attachments/assets/c7714b0b-52ed-4a42-a73c-b037b54e7609" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="755" height="574" alt="image" src="https://github.com/user-attachments/assets/2c984390-e4d9-4423-bf37-b3f9f054fece" />


### New Sample Data Prediction

<img width="865" height="135" alt="image" src="https://github.com/user-attachments/assets/21d66d4e-ee8f-4d53-9bbb-c1ec63ac6fc2" />

## RESULT
Thus the Neural Network Regression Model is developed, trained and tested Successfully.
