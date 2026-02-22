from TripletTrain import training
from TripletTest import test

csvTrain = r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\Triplet\TripletCSV\train4.csv"
csvTest = r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\Triplet\TripletCSV\test4.csv"
trainData = r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\DataSet\train"
testData = r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\DataSet\test"
epochs = 100

if __name__ == "__main__":
    training(csvTrain, trainData, epochs)

    test(csvTest, testData, threshold=0.05)