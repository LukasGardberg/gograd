package main

import (
	"fmt"
	"github.com/LukasGardberg/gograd/base"
)
func main() {
	a := base.Value(2.0)
	b := base.Value(1.0)
	c := a.Mul(b)
	d := c.Add(a)
	e := d.Pow(2.0)

	fmt.Printf("b: %v, c: %v\n", b, c)

	e.Backward()

	base.Show_graph(e)
}
