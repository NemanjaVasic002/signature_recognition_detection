import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataSet import SignatureDataset, transform
from SiameseNN import SiameseSignatureNet


def training(csvFile, datasetLocation, totalEpochs):
    signatureDataset = SignatureDataset(csvFile=csvFile, rootDirectory=datasetLocation, imageTransform=transform)
    trainLoader = DataLoader(signatureDataset, batch_size=4, shuffle=True)

    siameseModel = SiameseSignatureNet()
    computingDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siameseModel.to(computingDevice)

    lossCriterion = nn.BCELoss()
    modelOptimizer = optim.Adam(siameseModel.parameters(), lr=0.001)

    lossHistory = []

    siameseModel.train()
    for epoch in range(totalEpochs):
        runningLoss = 0.0
        for imageOne, imageTwo, batchLabels in trainLoader:
            imageOne, imageTwo, batchLabels = imageOne.to(computingDevice), \
                                              imageTwo.to(computingDevice), \
                                              batchLabels.to(computingDevice).unsqueeze(1)

            modelOptimizer.zero_grad()
            modelOutputs = siameseModel(imageOne, imageTwo)
            currentLoss = lossCriterion(modelOutputs, batchLabels)
            currentLoss.backward()
            modelOptimizer.step()

            runningLoss += currentLoss.item()

        averageLoss = runningLoss / len(trainLoader)
        lossHistory.append(averageLoss)
        print(f"Epoch {epoch + 1}/{totalEpochs}, Loss: {averageLoss}")

    with open("T5E200B4.csv", "w") as f:
        f.write("Epoha,Gubitak\n")
        for i, value in enumerate(lossHistory):
            f.write(f"{i+1},{value}\n")

    torch.save(siameseModel.state_dict(), "siamese_model.pth")