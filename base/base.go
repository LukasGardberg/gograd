package base

import (
	"math"
	"fmt"
)

/* Only accept float64 values?

Maybe accept both float64 and int in the future?
https://go.dev/doc/tutorial/generics#add_generic_function

*/

/* Store a float value and gradient */
type value struct {
	Val float64
	grad float64
	_backward func()
	// can't use 'value' as key in a map if we have a function field
	_prev []value
}


func Value(x float64) value {
	return value{Val: x, grad: 0.0}
}


func (a value) Print() {
	fmt.Println(a.Val)
}


func (a value) Add(b value) value {
	out := Value(a.Val + b.Val)
	out._prev = append(out._prev, a, b)

	out._backward = func() {
		a.grad += out.grad
		b.grad += out.grad
	}

	return out
}


func (a value) Sub(b value) value {
	out := a.Add(Value(-b.Val))
	return out
}


func (a value) Mul(b value) value {
	out := Value(a.Val * b.Val)
	out._prev = append(out._prev, a, b)

	out._backward = func() {
		a.grad += b.Val * out.grad
		b.grad += a.Val * out.grad
	}

	return out
}


func (a value) Pow(b value) value {
	out := Value(math.Pow(a.Val, b.Val))
	out._prev = append(out._prev, a)

	out._backward = func() {
		a.grad += b.Val * math.Pow(a.Val, b.Val - 1) * out.grad
	}

	return out
}


func (a value) Div(b value) value {
	out := b.Pow(Value(-1.0)).Mul(a)
	return out
}


func (a value) Backward() float64 {
	a.grad = 1.0
	fmt.Printf("Called backward on %v\n", a)

	// Hm, might have to do some stuff with pointers here...
	// Not getting right result
	a._backward()
	return a.grad
}
