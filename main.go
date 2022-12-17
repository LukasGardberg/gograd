package main

import (
	"github.com/LukasGardberg/gograd/base"
	"fmt"
)
func main() {
	a := base.Value(2.0)
	b := base.Value(3.0)
	c := a.Pow(&b)
	c.Backward()

	c.Print()
	fmt.Println("dc/dc:", c.Grad)
	fmt.Println("dc/da:", a.Grad)
	fmt.Println("dc/db:", b.Grad)
	fmt.Println()
}
