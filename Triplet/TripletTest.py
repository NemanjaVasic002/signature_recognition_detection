import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TripletNN import TripletSignatureNet
from TripletDataSet import SignatureDataset, transform


def test(csvFile, datasetLocation, threshold=1.0):
    testDataset = SignatureDataset(csvFile=csvFile, rootDirectory=datasetLocation, imageTransform=transform)
    # Batch size 1 je najbolji za preglednost rezultata po svakom potpisu
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

    Model = TripletSignatureNet()
    computingDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model.load_state_dict(torch.load("triplet_model.pth", map_location=computingDevice, weights_only=True))
    Model.to(computingDevice)
    Model.eval()

    correct_triplets = 0
    total_triplets = 0

    print(f"{'Red':<4} | {'Dist (A-P)':<12} | {'Dist (A-N)':<12} | {'Status'}")
    print("-" * 50)

    with torch.no_grad():
        for i, (anchor, positive, negative) in enumerate(testLoader):
            anchor = anchor.to(computingDevice)
            positive = positive.to(computingDevice)
            negative = negative.to(computingDevice)

            # Generisanje embedding-a za sve tri slike
            embA = Model.forwardOnce(anchor)
            embP = Model.forwardOnce(positive)
            embN = Model.forwardOnce(negative)

            # Računanje obe distance
            distAP = F.pairwise_distance(embA, embP).item()
            distAN = F.pairwise_distance(embA, embN).item()

            # Triplet je "ispravan" ako je original bliži ankoru nego falsifikat
            # i ako su obe distance u skladu sa pragom (threshold)
            is_correct = distAP < threshold and distAN > threshold

            if is_correct:
                correct_triplets += 1
                status = "OK"
            else:
                status = "FAIL"

            total_triplets += 1

            print(f"{i + 1:<4} | {distAP:<12.4f} | {distAN:<12.4f} | {status}")

    accuracy = (correct_triplets / total_triplets) * 100
    print("-" * 50)
    print(f"Total Triplet Accuracy: {accuracy:.2f}%")