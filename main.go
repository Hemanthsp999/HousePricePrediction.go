package main

import (
	"fmt"
	"log"
	"os"

	prediction "main.go/Prediction"
)

func main() {
	fmt.Println("................. Multiple Linear Regression Model .................")

	Model := prediction.Beta{}

	getFile, err := Model.LoadCsv("Dataset/ADANIPORTS.csv")
	if err != nil {
		log.Fatalf("Error while opening the file : %v", err)
		os.Exit(1)
	}

	// EDA
	Model.SummaryStats(getFile)

	// Split the dataset into Train and Test part
	trainRecords, testRecords := Model.SplitDataset(getFile, 10, 8)

	x1Train, x2Train, x3Train, yTrain := Model.PrepareData(trainRecords)

	x1Test, x2Test, x3Test, yTest := Model.PrepareData(testRecords)

	// Normalize the variables for efficiency
	x1Train = Model.Normalize(x1Train)
	x2Train = Model.Normalize(x2Train)
	x3Train = Model.Normalize(x3Train)
	x1Test = Model.Normalize(x1Test)
	x2Test = Model.Normalize(x2Test)
	x3Test = Model.Normalize(x3Test)

	var trainData, testData prediction.Mat

	trainData.X1, trainData.X2, trainData.X3 = x1Train, x2Train, x3Train

	testData.X1, testData.X2, testData.X3 = x1Test, x2Test, x3Test

	fmt.Println()

	// Fit the model
	Model.FitModel(trainData, yTrain)

	// Predict the TrainSet
	PredictedTrain, err := Model.PredictModel(trainData, yTrain)
	if err != nil {
		log.Fatalf("Error encountered %v", err)
	}
	// Predict the TestSet
	PredictedTest, err := Model.PredictModel(testData, yTest)
	if err != nil {
		log.Fatalf("Error encountered %v", err)
	}

	// Plots the Train Prediction
	Model.PlotGraph(yTrain, PredictedTrain, "Train.png")
	PredictedModel := PredictedTest
	for i := range PredictedTest {
		fmt.Printf("%.2f \t%.2f\n", yTest[i], PredictedTest[i])
	}

	// Plots the Test Prediction
	Model.PlotGraph(yTest, PredictedModel, "Test.png")
	Error1, Error2 := Model.Error(yTest, PredictedModel)

	r2_score, adj_R2_score, err := Model.Rsquare(yTest, PredictedModel)
	if err != nil {
		fmt.Println("Performance error", err)
	}

	fmt.Println("R2 Score( MODEL PERFORMANCE ) :", r2_score, "%", "\t\tAdjusted R2 score :", adj_R2_score)

	cost := Model.CostFunction(yTest, PredictedModel)
	fmt.Println("Cost Function : ", cost)

	fmt.Println("RMSE :", Error1, "\tMSE :", Error2)

}
