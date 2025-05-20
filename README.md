# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

</br>
Construct a binary classification model leveraging a pretrained VGG19 architecture to differentiate between defected and non-defected capacitors by modifying the final layer to a single neuron. Train the model using a dataset comprising images of capacitors with and without defects to enhance detection accuracy. Optimize and assess the model to ensure robust performance in capacitor quality assessment for manufacturing applications.
</br>
</br>

## DESIGN STEPS
### STEP 1:
Gather and preprocess a dataset containing images of defected and non-defected capacitors, ensuring proper data augmentation and normalization.

</br>

### STEP 2:
Divide the dataset into training, validation, and test sets to facilitate model evaluation and prevent overfitting.
</br>

### STEP 3:
Load the pretrained VGG19 model, initialized with ImageNet weights, to leverage its feature extraction capabilities.
<br>

### STEP 4:
Modify the architecture by removing the original fully connected layers and replacing the final output layer with a single neuron using a Sigmoid activation function for binary classification.
<br>

### STEP 5:
Train the model using the binary cross-entropy loss function and Adam optimizer, iterating through multiple epochs for optimal learning.

<br>

### STEP 6:
Assess performance by evaluating test data, analyzing key metrics such as the confusion matrix and classification report to measure accuracy and reliability in capacitor defect detection.
<br>


## PROGRAM
Include your code here
```python

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match one binary classes
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Tanushree A")
    print("Register Number: 212223100057")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()






```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2025-05-20 203638](https://github.com/user-attachments/assets/655bf973-043f-4eb0-8402-c45150993119)


### Confusion Matrix
![Screenshot 2025-05-20 203737](https://github.com/user-attachments/assets/cf5ae55b-9c68-472a-8279-feaf9fd62271)

### Classification Report
![Screenshot 2025-05-20 203805](https://github.com/user-attachments/assets/0959f329-1796-46d0-95eb-a05e4f820f1b)


### New Sample Prediction
![Screenshot 2025-05-20 203830](https://github.com/user-attachments/assets/ef754226-b3b0-43d1-bbc0-42b042ababff)

![Screenshot 2025-05-20 203850](https://github.com/user-attachments/assets/7282f9cd-eb20-4f11-9a95-ce3f8912cd0a)



## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors


</br>
</br>
