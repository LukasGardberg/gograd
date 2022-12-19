package main

import (
	"github.com/LukasGardberg/gograd/base"
	"fmt"
)
func main() {
	a := base.Value(2.0)
	b := base.Value(3.0)
	c := base.Value(4.0)
	d := ((a.Pow(2.0).Mul(b).Add(c)).Div(b)).Pow(2.0)
	topo := d.Backward()

	base.Show_graph(topo)

	fmt.Println(len(topo))
}

func test_1() {
	a := base.Value(2.0)
	b := base.Value(3.0)
	c := base.Value(4.0)
	d := ((a.Pow(2.0).Mul(b).Add(c)).Div(b)).Pow(2.0)
	d.Backward()

	d.Print()
	fmt.Println("dd/dd:", d.Grad)
	fmt.Println("dd/da:", a.Grad)
	fmt.Println("dd/db:", b.Grad)
	fmt.Println("dd/dc:", c.Grad)
	fmt.Println()
}