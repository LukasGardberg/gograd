Created new go module with 'go mod init gograd'

Todo:

- Make sense of pointers used in operations so that gradients are tracked properly
- add topological sort to backwards function

When we call `a.Backward()` on a `Value` struct, we want to calculate the derivative of `a` with respect to itself and all of its children, i.e. all `Values` (nodes) used in calculating `a`.