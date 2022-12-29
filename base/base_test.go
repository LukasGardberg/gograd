package base

import (
	"testing"

	ad "github.com/pbenner/autodiff"
)

// Use for test comparisons?
// https://github.com/pbenner/autodiff

// Test creating a value works
func TestValue(t *testing.T) {
	val := Value(1.0)
	if val.Val != 1.0 {
		t.Errorf("Expected value to be 1.0, got %f", val.Val)
	}
}

// Test adding two values works
func TestAdd(t *testing.T) {
	a := Value(1.0)
	b := Value(2.0)
	c := a.Add(b)
	if c.Val != 3.0 {
		t.Errorf("Expected value to be 3.0, got %f", c.Val)
	}
}

// Test subtracting two values works
func TestSub(t *testing.T) {
	a := Value(1.0)
	b := Value(2.0)
	c := a.Sub(b)
	if c.Val != -1.0 {
		t.Errorf("Expected value to be -1.0, got %f", c.Val)
	}
}

// Test multiplying two values works
func TestMul(t *testing.T) {
	a := Value(3.0)
	b := Value(2.0)
	c := a.Mul(b)
	if c.Val != 6.0 {
		t.Errorf("Expected value to be 6.0, got %f", c.Val)
	}
}

// Test raising a value to a power works
func TestPow(t *testing.T) {
	a := Value(3.0)
	c := a.Pow(2.0)
	if c.Val != 9.0 {
		t.Errorf("Expected value to be 9.0, got %f", c.Val)
	}
}

// Test dividing two values works
func TestDiv(t *testing.T) {
	a := Value(3.0)
	b := Value(2.0)
	c := a.Div(b)
	if c.Val != 1.5 {
		t.Errorf("Expected value to be 1.5, got %f", c.Val)
	}
}

// Test backpropagation works
func TestBackward(t *testing.T) {
	a := Value(3.0)
	b := Value(2.0)
	c := a.Mul(b)
	c.Backward()
	if a.Grad != 2.0 {
		t.Errorf("Expected gradient to be 1.0, got %f", a.Grad)
	}
	if b.Grad != 3.0 {
		t.Errorf("Expected gradient to be 1.0, got %f", b.Grad)
	}
}

func TestComparison(t *testing.T) {
	// compare derivatives of a * b

	// autodiff
	a := ad.NewReal64(2.0)
	b := ad.NewReal64(3.0)
	ad.Variables(1, a, b)
	a.Mul(a, b)

	ad_res1 := a.GetDerivative(0)
	ad_res2 := a.GetDerivative(1)

	// gograd
	c := Value(2.0)
	d := Value(3.0)
	e := c.Mul(d)
	e.Backward()

	gograd_res1 := c.Grad
	gograd_res2 := d.Grad

	if ad_res1 != gograd_res1 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res1, gograd_res1)
	}
	if ad_res2 != gograd_res2 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res2, gograd_res2)
	}
}

// Compare a more complex derivative
func TestComparison2(t *testing.T) {
	// compare derivatives of (a * b)^2

	// autodiff
	a := ad.NewReal64(2.0)
	b := ad.NewReal64(3.0)
	ad.Variables(1, a, b)
	a.Mul(a, b)
	a.Pow(a, ad.NewConstFloat64(2.0))

	ad_res1 := a.GetDerivative(0)
	ad_res2 := a.GetDerivative(1)

	// gograd
	c := Value(2.0)
	d := Value(3.0)
	e := c.Mul(d)
	f := e.Pow(2.0)
	f.Backward()

	gograd_res1 := c.Grad
	gograd_res2 := d.Grad

	if ad_res1 != gograd_res1 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res1, gograd_res1)
	}
	if ad_res2 != gograd_res2 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res2, gograd_res2)
	}
}

// Compare a more complex derivative
func TestComparison3(t *testing.T) {
	// compare derivatives of a^3 * 2 - b/3

	// autodiff
	a := ad.NewReal64(2.0)
	b := ad.NewReal64(3.0)
	ad.Variables(1, a, b)

	a.Pow(a, ad.NewConstFloat64(3.0))
	a.Mul(a, ad.NewConstFloat64(2.0))
	b.Div(b, ad.NewConstFloat64(3.0))
	a.Sub(a, b)

	ad_res1 := a.GetDerivative(0)
	ad_res2 := a.GetDerivative(1)

	// gograd
	a2 := Value(2.0)
	b2 := Value(3.0)
	a3 := a2.Pow(3.0)
	a32 := a3.Mul(Value(2.0))
	b3 := b2.Div(Value(3.0))
	h := a32.Sub(b3)

	h.Backward()

	gograd_res1 := a2.Grad
	gograd_res2 := b2.Grad

	// check forward pass
	if a.Value != h.Val {
		t.Errorf("Expected value to be %f, got %f", a.Value, h.Val)
	}

	// check gradients
	if ad_res1 != gograd_res1 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res1, gograd_res1)
	}
	if ad_res2 != gograd_res2 {
		t.Errorf("Expected gradient to be %f, got %f", ad_res2, gograd_res2)
	}

}

// Test print
func TestPrint(t *testing.T) {
	a := Value(2.0)
	b := Value(3.0)
	c := a.Mul(b)
	c.Print()
}

// Test show graph
func TestShowGraph(t *testing.T) {
	a := Value(2.0)
	b := Value(3.0)
	c := a.Mul(b)
	c.Backward()
	Show_graph(c)
}
