package main

import (
	"fmt"
	//. "github.com/LukasGardberg/gograd/base" // Like Python's "from base import *"
	ad "github.com/pbenner/autodiff"
)
func main() {
	a := ad.NewReal64(2.0)
	b := ad.NewReal64(3.0)
	ad.Variables(1, a, b)

	a.Mul(a, b)
	fmt.Println(a)
	fmt.Println(a.GetDerivative(0))
	
}