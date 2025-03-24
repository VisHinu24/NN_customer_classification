# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/4f9bc0c6-5a64-40c0-99c6-182303e1124e)


## DESIGN STEPS

## Step 1: Data Preprocessing  
- Clean and normalize the dataset.  
- Split data into training, validation, and test sets.  

## Step 2: Model Architecture  
- **Input Layer:** Number of neurons equal to the number of features.  
- **Hidden Layers:** Two layers with ReLU activation.  
- **Output Layer:** Four neurons (Segments A, B, C, D) with softmax activation.  

## Step 3: Model Compilation  
- Use **categorical crossentropy** as the loss function.  
- Optimize using the **Adam optimizer**.  
- Track **accuracy** as a performance metric.  

## Step 4: Training  
- Train the model using **early stopping** to prevent overfitting.  
- Use a **batch size** of 32 and set an appropriate number of epochs.  

## Step 5: Model Evaluation  
- Compile the model using the same **loss function**, **optimizer**, and **accuracy metric**.  

## Step 6: Final Training  
- Train the model again with early stopping, a **batch size of 32**, and a suitable number of epochs.  

## PROGRAM

### Name: Vishinu H 
### Register Number: 212223220124

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/4feaeb79-afb7-41a8-aa04-02ef268e6c5b)

## OUTPUT



### Confusion Matrix

![image](https://github.com/user-attachments/assets/5d16a75f-4e1e-47a4-ac9e-f31eb6227da5)


### Classification Report

![image](https://github.com/user-attachments/assets/2a0c9d7f-74b9-47bd-8ab8-42df50b62f2d)



### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/3c51e3da-47b7-417c-9be6-cce2ff1cd7ff)


## RESULT

Thus, we have developed a neural network classification model for the given dataset.
