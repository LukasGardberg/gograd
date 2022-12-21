package base

import (
	"math"
	"fmt"
	"github.com/dominikbraun/graph"
	"os"
	"github.com/google/uuid"
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
	_prev map[*value]bool
	_op string
	_id string
}


func Value(x float64) *value {
	uuid := uuid.NewString()
	return &value{Val: x, Grad: 0.0, _backward: func (out *value) {}, _prev: make(map[*value]bool), _id: uuid}
}


func (a *value) Print() {
	fmt.Println(a.Val)
}


func (a *value) Add(b *value) *value {
	out := Value(a.Val + b.Val)
	out._op = "+"
	out._prev[a], out._prev[b] = true, true

	out._backward = func (out *value) () {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}


func (a *value) Sub(b *value) *value {
	out := Value(a.Val - b.Val)
	out._op = "-"
	out._prev[a], out._prev[b] = true, true

	out._backward = func (out *value) () {
		a.Grad += out.Grad
		b.Grad -= out.Grad
	}

	return out
}


func (a *value) Mul(b *value) *value {
	out := Value(a.Val * b.Val)
	out._op = "*"
	out._prev[a], out._prev[b] = true, true
	
	// In the case of c = a * b, c will here be 'out'
	// so we need to make sure it's gradient is set to 1 in '_backward'
	out._backward = func (out *value) () {
		a.Grad += b.Val * out.Grad
		b.Grad += a.Val * out.Grad
	}

	return out
}

// only floats for now, how would gradient update work for 'value's?
func (a *value) Pow(b float64) *value {
	// 'pow' currently not accounted for in gradient caluculation
	pow := Value(b)

	out := Value(math.Pow(a.Val, b))
	out._op = "**"
	out._prev[a] = true
	out._prev[pow] = true // add to prev for node in graph viz

	out._backward = func (out *value) () {
		a.Grad += b * math.Pow(a.Val, b - 1) * out.Grad
	}

	return out
}


func (a *value) Div(b *value) *value {
	out := Value(a.Val / b.Val)
	out._op = "/"
	out._prev[a], out._prev[b] = true, true

	out._backward = func (out *value) () {
		a.Grad += 1.0 / b.Val * out.Grad
		b.Grad += -1.0 * a.Val / math.Pow(b.Val, 2) * out.Grad
	}

	return out
}


func (a *value) Backward() {
	a.Grad = 1.0
	topo := build_topo(a)
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._backward(topo[i])
	}
}


func build_topo (a *value) []*value {
	var topo []*value
	var seen = make(map[*value]bool)

	var build func (a *value)
	build = func (a *value) {
		if !seen[a] {
			seen[a] = true
			for v := range a._prev {
				build(v)
			}
			topo = append(topo, a)
		}
	}

	build(a)
	return topo
}


func Show_graph(root *value) {

	nodes, edges := get_nodes_edges(root)
	g := graph.New(graph.StringHash, graph.Directed(), graph.Acyclic())

	for k := range nodes {
		label := fmt.Sprintf("v: %.2f, grad: %.2f", k.Val, k.Grad)
		g.AddVertex(k._id, graph.VertexAttribute("label", label))

		if k._op != "" {
			g.AddVertex(k._id + k._op, graph.VertexAttribute("label", k._op))
			g.AddEdge(k._id + k._op, k._id)
		}
	}

	for k := range edges {
		a, b := (*k)[0], (*k)[1]
		g.AddEdge(a._id, b._id + b._op)
	}

	file, _ := os.Create("./my-graph.gv")
	_ = DOT(g, file)

}


func get_nodes_edges(root *value) (map[*value]bool, map[*[]*value]bool) {
	var nodes = make(map[*value]bool)
	var edges = make(map[*[]*value]bool)

	var build func (a *value)
	build = func (a *value) {
		if !nodes[a] {
			nodes[a] = true
			for v := range a._prev {
				e := []*value{v, a}
				edges[&e] = true
				build(v)
			}
		}
	}

	build(root)
	return nodes, edges
}