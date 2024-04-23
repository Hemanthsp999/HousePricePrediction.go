package predection

import (
	"math"
)

// Based on these values we've to predict the output
/*
type sPerformance struct {
	Percentage          float64
	TotalReturn         float64
	RelativePerformance float64
	EarningPerShare     float64
	PriceToEarnings     float64
	DividendYeild       float64
}
*/

type Linear struct {
	slope     float64
	intercept float64
	Y         []float64
}

var L Linear

func (L *Linear) FitModel(x []float64, y []float64) {
	// Important factors ? important is MeanX MeanY sum(x) sum(y)

	n := len(x)

	// using "make" cause we're using slices, if its structs then we can go for "new"
	var MeanX, MeanY float64

	var sumX, sumY, numerator, denominator float64

	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
	}

	// Finding " Mean of X and Y"
	MeanX = sumX / float64(n)
	MeanY = sumY / float64(n)

	for i := 0; i < n; i++ {
		numerator += ((x[i] - MeanX) * (y[i] - MeanY))
		denominator += math.Pow((x[i] - MeanX), 2)
	}

	// Linear Regression Formula  y = mx + c
	// Where m = slope and it's formula  m = ((sumX - MeanX) * (sumY - MeanY))/(sumX - MeanX)^2
	// and c = intercept means where x and y at which point they both meets
	// c = MeanY - m * MeanX

	L.slope = numerator / denominator

	L.intercept = MeanY - L.slope*MeanX

}

func (L *Linear) ModelPrediction(x []float64) []float64 {

	n := len(x)

	L.Y = make([]float64, n)
	for i := 0; i < n; i++ {
		// where y is predicted value
		L.Y[i] = L.slope*x[i] + L.intercept
	}

	return L.Y
}

func (L *Linear) StockModel(data [][]float64) {
	n := len(data)

	if n < 2 {
		panic("not enough length")
	}

	openPrice := make([]float64, n)
	closePrice := make([]float64, n)

	for i, v := range data {
		openPrice[i] = v[0]
		closePrice[i] = v[1]
	}

	L.FitModel(openPrice, closePrice)

}

/*	percentage = ((curPrice - initPrice) / initPrice) * 100
	totalReturn := ((endValue - (begValue + dividends)) / begValue) * 100
	relativePerformance := (stockReturn / benchMarkReturn) * 100
	earnPerShare := netIncome / nSharesOutstanding
	priceToEarnings := marketPricePerShare / earningsPerShare
	dividendYeild := (annualDividendPerShare / stockPrice) * 100
	var curPrice float64
	var initPrice float64
	var endValue float64
	var begValue float64
	var dividends float64
	var stockReturn float64
	var benchMarkReturn float64
	var netIncome float64
	var nSharesOutstanding float64
	var marketPricePerShare float64
	var earningsPerShare float64
	var annualDividendPerShare float64
	var stockPrice float64
	var err error


*/
