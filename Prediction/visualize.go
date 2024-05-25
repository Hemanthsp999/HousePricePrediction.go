package prediction

import (
	"fmt"
	"image/color"
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func (M *Beta) PlotGraph(data []float64, y []float64, filePath string) {
	plot := plot.New()

	newplot, err := plotter.NewScatter(getPlotPoints(data, y))
	if err != nil {
		log.Fatalf("Error while graph writing %v\n", err)
	}
	//newplot.Color = color.RGBA{R: 255,A: 255}

	newplot.GlyphStyle.Shape = draw.CircleGlyph{}
	newplot.GlyphStyle.Radius = vg.Points(3)
	newplot.GlyphStyle.Color = color.RGBA{R: 255, A: 255}

	plot.Add(newplot)

	if err := plot.Save(512, 512, filePath); err != nil {
		fmt.Println("Error", err)
	}
}

func getPlotPoints(data, y []float64) plotter.XYs {
	plot := make(plotter.XYs, len(data))

	for i := range data {
		plot[i].X = data[i]
		plot[i].Y = y[i]
	}

	return plot
}
