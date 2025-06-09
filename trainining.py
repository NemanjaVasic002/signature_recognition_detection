from datasetsCreation import *
from nn import *

def training(csvFile,datasetLoc,numberOfTrainingExamples):

#resizeSignatures(r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\Nemanja", r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\NResize", target_height=512,targetWidth= 512)
    dataset = SignatureDataset(csv_file=csvFile,root_dir=datasetLoc, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SignatureCNN()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(numberOfTrainingExamples):  # Change number of epochs as needed
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            print("Labels:", labels)


            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Add dimension for BCELoss

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")