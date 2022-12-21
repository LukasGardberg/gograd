package base

import (
	"testing"
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