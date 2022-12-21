package main

import (
	base "github.com/LukasGardberg/gograd/base"
	"fmt"
)
func main() {
	a := base.Value(2.0)
	b := base.Value(1.0)
	c := a.Mul(b)

	fmt.Printf("b: %v, c: %v\n", b, c)

	c.Backward()

	base.Show_graph(c)
}
