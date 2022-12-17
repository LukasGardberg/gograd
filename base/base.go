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
	Grad float64
	_backward func (out *value) ()
	// can't use 'value' as key in a map if we have a function field
	_prev []*value
}


func Value(x float64) value {
	return value{Val: x, Grad: 0.0}
}


func (a *value) Print() {
	fmt.Println(a.Val)
}


func (a *value) Add(b *value) value {
	out := Value(a.Val + b.Val)
	out._prev = append(out._prev, a, b)

	out._backward = func (out *value) () {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}


func (a *value) Sub(b *value) value {
	out := Value(a.Val - b.Val)
	out._prev = append(out._prev, a, b)

	out._backward = func (out *value) () {
		a.Grad += out.Grad
		b.Grad -= out.Grad
	}

	return out
}


func (a *value) Mul(b *value) value {
	out := Value(a.Val * b.Val)
	out._prev = append(out._prev, a, b)
	
	// In the case of c = a * b, c will here be 'out'
	// so we need to make sure it's gradient is set to 1 in '_backward'
	out._backward = func (out *value) () {
		a.Grad += b.Val * out.Grad
		b.Grad += a.Val * out.Grad
	}

	return out
}

// wrong for d/db (a^b) = a^b * log(a) ?
func (a *value) Pow(b *value) value {
	out := Value(math.Pow(a.Val, b.Val))
	out._prev = append(out._prev, a)

	out._backward = func (out *value) () {
		a.Grad += b.Val * math.Pow(a.Val, b.Val - 1) * out.Grad
		
	}

	return out
}


func (a *value) Div(b *value) value {
	out := Value(a.Val / b.Val)
	out._prev = append(out._prev, a, b)

	out._backward = func (out *value) () {
		a.Grad += 1.0 / b.Val * out.Grad
		b.Grad += -1.0 * a.Val / math.Pow(b.Val, 2) * out.Grad
	}

	return out
}


func (a *value) Backward() {
	a.Grad = 1.0
	fmt.Printf("Called backward on %v\n", a)

	// Hm, might have to do some stuff with pointers here...
	// Not getting right result
	a._backward(a)
}
