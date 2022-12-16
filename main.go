package main

import (
	"github.com/LukasGardberg/gograd/base"
	"fmt"
)
func main() {
	a := base.Value(2.0)
	b := base.Value(3.0)

	fmt.Println(a.Mul(b).Backward())
}
