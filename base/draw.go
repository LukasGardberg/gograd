// ** Adopted from https://github.com/dominikbraun/graph/blob/main/draw/draw.go **

package base

import (
	"fmt"
	"io"
	"os"
	"text/template"

	"github.com/dominikbraun/graph"
)

// const dotTemplate = `strict {{.GraphType}} {
// {{range $s := .Statements}}
// 	"{{.Source}}" {{if .Target}}{{$.EdgeOperator}} "{{.Target}}" [ {{range $k, $v := .EdgeAttributes}}{{$k}}="{{$v}}", {{end}} weight={{.EdgeWeight}} ]{{else}}[ {{range $k, $v := .SourceAttributes}}{{$k}}="{{$v}}", {{end}} weight={{.SourceWeight}} ]{{end}};
// {{end}}
// }
// `

// Added LR direction, record node shape
const dotTemplate = `strict {{.GraphType}} { 
	rankdir=LR; {{range $s := .Statements}} 
	"{{.Source}}" {{if .Target}}{{$.EdgeOperator}} "{{.Target}}" [ {{range $k, $v := .EdgeAttributes}}{{$k}}="{{$v}}, {{end}} weight={{.EdgeWeight}} ]{{else}}[ {{range $k, $v := .SourceAttributes}}{{$k}}="{{$v}}", {{end}} shape=record, weight={{.SourceWeight}} ]{{end}}; 
	{{end}} }
	`

type description struct {
	GraphType    string
	EdgeOperator string
	Statements   []statement
}

type statement struct {
	Source           interface{}
	Target           interface{}
	SourceWeight     int
	SourceAttributes map[string]string
	EdgeWeight       int
	EdgeAttributes   map[string]string
}

func DOT[K comparable, T any](g graph.Graph[K, T], w io.Writer) error {
	desc, err := generateDOT(g)
	if err != nil {
		return fmt.Errorf("failed to generate DOT description: %w", err)
	}

	return renderDOT(w, desc)
}

func generateDOT[K comparable, T any](g graph.Graph[K, T]) (description, error) {
	desc := description{
		GraphType:    "graph",
		EdgeOperator: "--",
		Statements:   make([]statement, 0),
	}

	if g.Traits().IsDirected {
		desc.GraphType = "digraph"
		desc.EdgeOperator = "->"
	}

	adjacencyMap, err := g.AdjacencyMap()
	if err != nil {
		return desc, err
	}

	for vertex, adjacencies := range adjacencyMap {
		_, sourceProperties, err := g.VertexWithProperties(vertex)
		if err != nil {
			return desc, err
		}

		stmt := statement{
			Source:           vertex,
			SourceWeight:     sourceProperties.Weight,
			SourceAttributes: sourceProperties.Attributes,
		}
		desc.Statements = append(desc.Statements, stmt)

		for adjacency, edge := range adjacencies {
			stmt := statement{
				Source:         vertex,
				Target:         adjacency,
				EdgeWeight:     edge.Properties.Weight,
				EdgeAttributes: edge.Properties.Attributes,
			}
			desc.Statements = append(desc.Statements, stmt)
		}
	}

	return desc, nil
}

func renderDOT(w io.Writer, d description) error {
	tpl, err := template.New("dotTemplate").Parse(dotTemplate)
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}

	return tpl.Execute(w, d)
}

func Show_graph(root *value) {

	nodes, edges := get_nodes_edges(root)
	g := graph.New(graph.StringHash, graph.Directed(), graph.Acyclic())

	for k := range nodes {
		label := fmt.Sprintf("v: %.2f, grad: %.2f", k.Val, k.Grad)
		g.AddVertex(k._id, graph.VertexAttribute("label", label))

		if k._op != "" {
			g.AddVertex(k._id+k._op, graph.VertexAttribute("label", k._op))
			g.AddEdge(k._id+k._op, k._id)
		}
	}

	for k := range edges {
		a, b := (*k)[0], (*k)[1]
		g.AddEdge(a._id, b._id+b._op)
	}

	file, _ := os.Create("./my-graph.gv")
	_ = DOT(g, file)

}

func get_nodes_edges(root *value) (map[*value]bool, map[*[]*value]bool) {
	var nodes = make(map[*value]bool)
	var edges = make(map[*[]*value]bool)

	var build func(a *value)
	build = func(a *value) {
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
