package base

import (
	"math"
	"fmt"
	"github.com/dominikbraun/graph"
	"strconv"
	"os"
	//"github.com/google/uuid"
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
	// _prev []*value
	_prev map[*value]bool
	_op string
	_id string
}

// A new uuid needs to be generated for each value, and to be used in
// building the compute graph
func Value(x float64) *value {
	return &value{Val: x, Grad: 0.0, _backward: func (out *value) {}, _prev: make(map[*value]bool)}
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
	out := Value(math.Pow(a.Val, b))
	out._op = "**"
	out._prev[a] = true

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
	fmt.Printf("Called backward on %v\n", a)

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


func val_hash(v *value) string {
	return strconv.FormatFloat(v.Val, 'f', 2, 64)
}


func Show_graph(root *value) {

	nodes, edges := get_nodes_edges(root)
	g := graph.New(graph.StringHash, graph.Directed(), graph.Acyclic())

	for k := range nodes {
		label := fmt.Sprintf("v: %.2f, grad: %.2f", k.Val, k.Grad)
		g.AddVertex(val_hash(k), graph.VertexAttribute("label", label))

		if k._op != "" {
			g.AddVertex(val_hash(k) + k._op, graph.VertexAttribute("label", k._op))
			g.AddEdge(val_hash(k) + k._op, val_hash(k))
		}
	}

	//add edges
	for k := range edges {
		a, b := (*k)[0], (*k)[1]
		g.AddEdge(val_hash(a), val_hash(b) + b._op)
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
				// add v and a to edges
				e := []*value{v, a}
				edges[&e] = true
				build(v)
			}
		}
	}

	build(root)
	return nodes, edges
}