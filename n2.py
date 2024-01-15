from enum import Enum
import time
import numpy as np
import networkx as nx
import random
from copy import deepcopy
import matplotlib.pyplot as plt


def steep_sigmoid_activation(z):
    y = 1.0 / (1.0 + np.exp(-4.924273 * z))
    return float(y)


class NodeTypes(Enum):
    Input = 0
    Output = 1
    Hidden = 2
    Bias = 3


class NodeGene:
    marking = 0

    def __init__(self, t: NodeTypes, activation=steep_sigmoid_activation) -> None:
        self.type = t
        self.marking = NodeGene.marking
        NodeGene.marking += 1
        self.activation = activation
        self.acts = 0
        if self.type == NodeTypes.Bias:
            self.sum = 1
        else:
            self.sum = 0

    @staticmethod
    def copy(self):
        return deepcopy(self)

    def reset(self):
        if self.type == NodeTypes.Bias:
            self.sum = 1
        else:
            self.sum = 0
        self.acts = 0


class ConnectGene:
    marking = 0

    def __init__(
        self,
        in_node_marking: int,
        out_node_marking: int,
        weight: float,
        disabled: bool = False,
    ) -> None:
        self.in_node_marking = in_node_marking
        self.out_node_marking = out_node_marking
        self.weight = weight
        self.disabled = disabled
        self.marking = ConnectGene.marking
        ConnectGene.marking += 1

    @staticmethod
    def copy(self):
        return deepcopy(self)


class Individuum:
    def __init__(
        self, node_genes: list[NodeGene], connect_genes: list[ConnectGene]
    ) -> None:
        self.node_genes: list[NodeGene] = node_genes
        self.connect_genes: list[NodeGene] = sorted(connect_genes, key=lambda x: x.marking)
        self.fitness = 0
        self.input_nodes = sorted(
            [node for node in self.node_genes if node.type == NodeTypes.Input],
            key=lambda node: node.marking,
        )
        self.output_nodes = sorted(
            [node for node in self.node_genes if node.type == NodeTypes.Output],
            key=lambda node: node.marking,
        )
        self.bias_nodes = [
            node for node in self.node_genes if node.type == NodeTypes.Bias
        ]

    @staticmethod
    def copy(self):
        return deepcopy(self)

    def mutate(self, mutation: dict):
        # Mutate Weights
        if random.random() < mutation["weight"]["rate"]:
            for connection in self.connect_genes:
                if random.random() < mutation["weight"]["reset"]:
                    connection.weight = np.random.normal(0, 1)
                else:
                    connection.weight += np.random.normal(0, 0.1)
        # Mutate Connections
        if random.random() < mutation["connection"]:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < 1:
                in_node_marking = random.choice(
                    [
                        node.marking
                        for node in self.node_genes
                        if node.type != NodeTypes.Output
                    ]
                )
                out_node_marking = random.choice(
                    [
                        node.marking
                        for node in self.node_genes
                        if node.type != NodeTypes.Input and node.type != NodeTypes.Bias
                    ]
                )
                if in_node_marking == out_node_marking:
                    continue
                if not [
                    connect
                    for connect in self.connect_genes
                    if connect.in_node_marking == in_node_marking
                    and connect.out_node_marking == out_node_marking
                ]:
                    self.connect_genes.append(
                        ConnectGene(
                            in_node_marking, out_node_marking, np.random.normal(0, 1)
                        )
                    )
                    break
            self.connect_genes.sort(key=lambda connect: connect.marking)
        # Mutate Nodes
        if random.random() < mutation["node"]:
            connection = random.choice(
                [
                    connect
                    for connect in self.connect_genes
                    if not connect.disabled
                    and connect.in_node_marking
                    not in [node.marking for node in self.bias_nodes]
                ]
            )
            connection.disabled = True
            new_node = NodeGene(NodeTypes.Hidden)
            self.node_genes.append(new_node)
            self.node_genes.sort(key=lambda node: node.marking)
            self.connect_genes.append(
                ConnectGene(
                    connection.in_node_marking, new_node.marking, np.random.normal(0, 1)
                )
            )
            self.connect_genes.append(
                ConnectGene(
                    new_node.marking,
                    connection.out_node_marking,
                    np.random.normal(0, 1),
                )
            )
            self.connect_genes.sort(key=lambda connect: connect.marking)

    def distance(self, other: "Individuum", c1, c2, c3) -> float:
        # Distance between two Individuums
        unique_markings = list(
            set(
                [gene.marking for gene in self.connect_genes]
                + [gene.marking for gene in other.connect_genes]
            )
        )
        disjoint = 0
        weight_diff = []
        i,j=0,0

        for marking in sorted(unique_markings):
            if marking == self.connect_genes[i].marking and marking == other.connect_genes[j].marking:
                weight_diff.append(abs(
                    self.connect_genes[i].weight - other.connect_genes[j].weight
                ))
                i += 1
                j += 1
            elif marking == self.connect_genes[i].marking:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1
            
            if i >= len(self.connect_genes) or j >= len(other.connect_genes):
                break

            excess = i - (len(self.connect_genes)-1) + j - (len(other.connect_genes)-1)

        return (c1 * excess) / len(unique_markings) + (c2 * disjoint) / len(unique_markings) + c3 * sum(weight_diff)/len(weight_diff)

    def crossover(self, other: "Individuum") -> "Individuum":
        # Crossover Connection Genes
        unique_markings = list(
            set(
                [gene.marking for gene in self.connect_genes]
                + [gene.marking for gene in other.connect_genes]
            )
        )
        child_connect_genes = []
        i, j = 0, 0
        while True:
            if self.connect_genes[i].marking == other.connect_genes[j].marking:
                if random.random() < 0.5:
                    child_connect_genes.append(ConnectGene.copy(self.connect_genes[i]))
                else:
                    child_connect_genes.append(ConnectGene.copy(other.connect_genes[j]))
                i += 1
                j += 1
            elif self.connect_genes[i].marking < other.connect_genes[j].marking:
                if self.fitness > other.fitness:
                    child_connect_genes.append(ConnectGene.copy(self.connect_genes[i]))
                i += 1
            else:
                if other.fitness > self.fitness:
                    child_connect_genes.append(ConnectGene.copy(other.connect_genes[j]))
                j += 1
            if i >= len(self.connect_genes):
                if other.fitness > self.fitness:
                    child_connect_genes += [
                        ConnectGene.copy(gene) for gene in other.connect_genes[j:]
                    ]
                elif self.fitness == other.fitness:
                    child_connect_genes += [
                        ConnectGene.copy(gene)
                        for gene in other.connect_genes[j:]
                        if random.random() < 0.5
                    ]
                break
            if j >= len(other.connect_genes):
                if self.fitness > other.fitness:
                    child_connect_genes += [
                        ConnectGene.copy(gene) for gene in self.connect_genes[i:]
                    ]
                elif self.fitness == other.fitness:
                    child_connect_genes += [
                        ConnectGene.copy(gene)
                        for gene in self.connect_genes[i:]
                        if random.random() < 0.5
                    ]
                break
        child_nodes = [NodeGene.copy(node) for node in self.node_genes]
        child_nodes += [
            NodeGene.copy(node)
            for node in other.node_genes
            if node.marking not in [node.marking for node in child_nodes]
        ]
        return Individuum(child_nodes, child_connect_genes)

    def forward(self, inputs):
        next_layer = []
        for node, inp in zip(self.input_nodes, inputs):
            for connection in [
                connect
                for connect in self.connect_genes
                if connect.in_node_marking == node.marking and not connect.disabled
            ]:
                out_node = [
                    node
                    for node in self.node_genes
                    if node.marking == connection.out_node_marking
                ][0]
                out_node.sum += connection.weight * inp
                next_layer.append(out_node) if out_node not in next_layer else None
                out_node.acts += 1
        while next_layer:
            current_layer = next_layer
            for node in current_layer:
                if (
                    len(
                        [
                            connect
                            for connect in self.connect_genes
                            if connect.out_node_marking == node.marking
                            and not connect.disabled
                        ]
                    )
                    == node.acts
                ):
                    node.sum = node.activation(node.sum)
            next_layer = []
            for node in current_layer:
                for connection in [
                    connect
                    for connect in self.connect_genes
                    if connect.in_node_marking == node.marking and not connect.disabled
                ]:
                    out_node = [
                        node
                        for node in self.node_genes
                        if node.marking == connection.out_node_marking
                    ][0]
                    out_node.sum += connection.weight * node.sum
                    out_node.acts += 1
                    next_layer.append(out_node) if out_node not in next_layer else None
        ret = np.array([node.sum for node in self.output_nodes])
        for node in self.node_genes:
            node.reset()
        return ret

    def plot(self):
        G = nx.DiGraph()

        G.add_nodes_from(
            (node.marking, {"pos": (i, 1)}) for i, node in enumerate(self.input_nodes)
        )
        G.add_nodes_from(
            (node.marking, {"pos": (len(self.input_nodes), 1)})
            for node in self.bias_nodes
        )
        G.add_nodes_from(
            (node.marking, {"pos": (i, 0)})
            for i, node in enumerate(self.node_genes)
            if node.type == NodeTypes.Hidden
        )
        G.add_nodes_from(
            (node.marking, {"pos": (i, -1)}) for i, node in enumerate(self.output_nodes)
        )
        for edge in [
            connect for connect in self.connect_genes if connect.disabled == False
        ]:
            G.add_edge(edge.in_node_marking, edge.out_node_marking)

        edge_labels = {
            (connect.in_node_marking, connect.out_node_marking): str(connect.marking)
            for connect in self.connect_genes
            if connect.disabled == False
        }
        pos = nx.get_node_attributes(G, "pos")

        # Color nodes based on type (green for input, black for output, red for bias)
        node_colors = [
            "green"
            if node in [node.marking for node in self.input_nodes]
            else "black"
            if node in [node.marking for node in self.output_nodes]
            else "red"
            if node in [node.marking for node in self.bias_nodes]
            else "gray"
            for node in G.nodes
        ]

        nx.draw(
            G,
            pos,
            with_labels=False,
            arrows=True,
            node_size=700,
            font_size=8,
            node_color=node_colors,
            font_color="white",
            font_weight="bold",
            edge_color="black",
        )

        # Add edge labels directly on the edges
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="red",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.1"),
        )
        plt.show()

    def printsum(self):
        for node in self.node_genes:
            print(f"Node: {node.marking}")

        for connect in self.connect_genes:
            print(f"Connect: {connect.marking}")
            print(f"In Node: {connect.in_node_marking}")
            print(f"Out Node: {connect.out_node_marking}")
            print(f"Weight: {connect.weight}")
            print(f"Disabled: {connect.disabled}\n")
