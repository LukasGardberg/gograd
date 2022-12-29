package base

import (
	//"math"
	"math/rand"
)

type neuron struct {
	Weights []*value
	Bias    *value
	Nonlin  func(a *value) *value
}

func Neuron(n_in int) *neuron {
	weights := make([]*value, n_in)
	bias := Value(0.0)
	for i := 0; i < n_in; i++ {
		weights[i] = Value(rand.NormFloat64())
	}

	nonlin := func(a *value) *value {
		return a.Relu()
	}

	return &neuron{Weights: weights, Bias: bias, Nonlin: nonlin}
}

func (neuron *neuron) Forward(inputs []float64) *value {
	// Sum the inputs
	sum := Value(0.0)
	for i := 0; i < len(inputs); i++ {
		sum = sum.Add(neuron.Weights[i].Mul(Value(inputs[i])))
	}

	sum = sum.Add(neuron.Bias)

	return neuron.Nonlin(sum)
}
