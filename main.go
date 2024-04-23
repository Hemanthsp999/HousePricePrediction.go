package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"main.go/predection"
)

func main() {
	fmt.Println("Enterd the Stock dataset")
	fileRead, err := os.Open("Stock_Dataset.csv")
	if err != nil {
		err = fmt.Errorf("Error while opening the file,Please check the FilePath %v", err)
		return
	}

	defer fileRead.Close()

	reader := csv.NewReader(fileRead)

	// skip the header row

	_, err = reader.Read()
	if err != nil {
		panic(err)
	}

	// Read the remainging record

	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	var x, y []float64

	for _, record := range records {
		xVal, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			fmt.Println("Error", err)
			return
		}
		yVal, err := strconv.ParseFloat(record[6], 64)
		if err != nil {
			fmt.Println("Error", err)
			return
		}

		x = append(x, xVal)
		y = append(y, yVal)
	}

	predection.L.FitModel(x, y)

	predict := predection.L.ModelPrediction(x)

	for i,record := range records {
		fmt.Printf("%s       %.2f      %.2f\n",record[0], y[i], predict[i])
		
	}
}
