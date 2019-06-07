from Models import fastIsing
import networkx as nx, copy

graph = nx.path_graph(5)
model = fastIsing.Ising(graph = graph, temperature = 2)
print(dir(model ))#
tmp = {}
for k in dir(model):
    attr = getattr(model, k)
    if isinstance(attr, str):
        tmp[k] = attr
print(tmp)
for i in range(4):
    print(i)
    copied = fastIsing.Ising(graph = graph, temperature = 2)
    # copied = copy.deepcopy(model)
    print(copied)
    for k in dir(copied):
        x = getattr(copied, k)
        if isinstance(x, str):
            assert getattr(copied, k) == tmp.get(k, 'sanity')
