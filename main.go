package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"

	prediction "main.go/Prediction"
)

func main() {

	/*
		Now, this model is not Performing that well,
		reason could be mainly Data Scalable
		For that there 2 types
		1.Normalization 2.Standardization
		Normalization(it scale down value features between 0 and 1) := x = (x - xmin) / (xmax - xmin)
		Standardization := z = x - miu / sigma
		here mean = 0 and standard deviation = 1

	*/

	fmt.Println("Implementing Multiple Linear Regression through Stock for Predection")

	FileOpen, err := os.Open("Datasets/BPCL.csv")
	if err != nil {
		fmt.Println("Error while opening file", err)
		os.Exit(1)
	}

	reader := csv.NewReader(FileOpen)

	reader.Read()

	CsvFile, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error while reading the content", err)
		os.Exit(2)
	}

	// These are Independent Variable
	x1 := make([]float64, 0)
	x2 := make([]float64, 0)
	x3 := make([]float64, 0)
//	x4 := make([]float64, 0)

	// its a Dependent Variable
	y := make([]float64, 0)

	minX1 := math.Inf(1)
	maxX1 := math.Inf(-1)
	minX2 := math.Inf(1)
	maxX2 := math.Inf(-1)
	minX3 := math.Inf(1)
	maxX3 := math.Inf(-1)
	//minX4 := math.Inf(1)
	//maxX4 := math.Inf(-1)
	minY := math.Inf(1)
	maxY := math.Inf(-1)

	for _, column := range CsvFile {
		x1Val, err := strconv.ParseFloat(column[3], 64)
		if err != nil {
			fmt.Errorf("Error while reading Opening column")
			os.Exit(1)
		}
		x2Val, err := strconv.ParseFloat(column[4], 64)
		if err != nil {
			fmt.Errorf("Error while reading High column")
			os.Exit(1)
		}
		x3Val, err := strconv.ParseFloat(column[5], 64)
		if err != nil {
			fmt.Errorf("Error while reading Low column")
			os.Exit(1)
		}
		/*x4Val, err := strconv.ParseFloat(column[8], 64)
		if err != nil {
			fmt.Errorf("Error while reading Low column")
			os.Exit(1)
		}
		*/
		yVal, err := strconv.ParseFloat(column[7], 64)
		if err != nil {
			os.Exit(1)
		}

		x1 = append(x1, x1Val)
		x2 = append(x2, x2Val)
		x3 = append(x3, x3Val)
		y = append(y, yVal)
		// Update min and max values
		if x1Val < minX1 {
			minX1 = x1Val
		}
		if x1Val > maxX1 {
			maxX1 = x1Val
		}
		if x2Val < minX2 {
			minX2 = x2Val
		}
		if x2Val > maxX2 {
			maxX2 = x2Val
		}
		if x3Val < minX3 {
			minX3 = x3Val
		}
		if x3Val > maxX3 {
			maxX3 = x3Val
		}
		/*
			if x4Val < minX4 {
				minX4 = x4Val
			}
			if x4Val > maxX4 {
				maxX4 = x4Val
			}
		*/
		if yVal < minY {
			minY = yVal
		}
		if yVal > maxY {
			maxY = yVal
		}

	}

	// Normalize the data
	for i := range x1 {
		x1[i] = (x1[i] - minX1) / (maxX1 - minX1)
		x2[i] = (x2[i] - minX2) / (maxX2 - minX2)
		x3[i] = (x3[i] - minX3) / (maxX3 - minX3)
		y[i] = (y[i] - minY) / (maxY - minY)
	}

	price := make([]float64, len(y))
	for i, _ := range y {
		price[i] = (y[i] / float64(len(y))) * 100
	}

	var data prediction.Mat

	data.X1 = x1
	data.X2 = x2
	data.X3 = x3

	var ModelFit prediction.Beta

	ModelFit.FitModel(data, y)

	PredictedModel, err := ModelFit.Prediction(data)
	if err != nil {
		fmt.Println("Error while predicting")
	}
	Error := ModelFit.RootMeanSquare(y, PredictedModel)
	for i, _ := range PredictedModel {
		fmt.Printf("Actual Closing Price  %.2f     Predicted Closing Price   %.2f    Error is %.2f\n", y[i], math.Round(PredictedModel[i]), Error[i])
	}

	Performance, err := ModelFit.Rsquare(y, PredictedModel)
	if err != nil {
		fmt.Println("Performance error", err)
	}
	fmt.Println(Performance)
}
