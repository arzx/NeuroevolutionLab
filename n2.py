from enum import Enum
import math
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

    def __init__(
        self, t: NodeTypes, marking, activation=steep_sigmoid_activation
    ) -> None:
        self.type = t
        self.marking = marking
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

    @staticmethod
    def rst_marking():
        NodeGene.marking = 0

    def reset(self):
        if self.type == NodeTypes.Bias:
            self.sum = 1
        else:
            self.sum = 0
        self.acts = 0


class ConnectGene:
    marking = 0
    innos = dict()

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
        if (in_node_marking, out_node_marking) in ConnectGene.innos:
            self.marking = ConnectGene.innos[(in_node_marking, out_node_marking)]
        else:
            self.marking = ConnectGene.marking
            ConnectGene.innos[(in_node_marking, out_node_marking)] = self.marking
            ConnectGene.marking += 1

    @staticmethod
    def copy(self):
        return deepcopy(self)

    @staticmethod
    def rst_marking():
        ConnectGene.marking = 0
        ConnectGene.innos = dict()


def reset_markings():
    NodeGene.rst_marking()
    ConnectGene.rst_marking()


class Individuum:
    def __init__(
        self, node_genes: list[NodeGene], connect_genes: list[ConnectGene]
    ) -> None:
        self.node_genes: list[NodeGene] = sorted(node_genes, key=lambda x: x.marking)
        self.connect_genes: list[NodeGene] = sorted(
            connect_genes, key=lambda x: x.marking
        )
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
            in_node_marking = random.choice(
                [
                    node.marking
                    for node in self.node_genes
                    if node.type != NodeTypes.Output
                    and len(
                        [
                            connect.out_node_marking
                            for connect in self.connect_genes
                            if connect.in_node_marking == node.marking
                        ]
                    )
                    < len(
                        [
                            node
                            for node in self.node_genes
                            if node.type == NodeTypes.Output
                            or node.type == NodeTypes.Hidden
                        ]
                    )
                ]
                + [node.marking for _ in range(5) for node in self.bias_nodes]
            )
            out_choice = [
                node.marking
                for node in self.node_genes
                if node.type != NodeTypes.Input
                and node.type != NodeTypes.Bias
                and node.marking != in_node_marking
            ]
            for marking in out_choice:
                if [
                    connect
                    for connect in self.connect_genes
                    if connect.in_node_marking == in_node_marking
                    and connect.out_node_marking == marking
                ]:
                    out_choice.remove(marking)
            G = nx.DiGraph()
            G.add_nodes_from([node.marking for node in self.node_genes])
            G.add_edges_from(
                [
                    (connect.in_node_marking, connect.out_node_marking)
                    for connect in self.connect_genes
                ]
            )
            rem = []
            for marking in out_choice:
                G.add_edge(in_node_marking, marking)
                if sum([1 for _ in nx.simple_cycles(G)]) != 0:
                    rem.append(marking)
                G.remove_edge(in_node_marking, marking)
            for marking in rem:
                out_choice.remove(marking)
            out_node_marking = random.choice(out_choice) if out_choice else None
            if out_node_marking is not None:
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
                    self.connect_genes.sort(key=lambda connect: connect.marking)

        # Mutate Nodes
        if random.random() < mutation["node"]:
            possible_connections = [
                connect
                for connect in self.connect_genes
                if not connect.disabled
                and connect.in_node_marking
                not in [node.marking for node in self.bias_nodes]
            ]
            connection = (
                random.choice(possible_connections) if possible_connections else None
            )
            if connection is not None:
                new_node = NodeGene(
                    NodeTypes.Hidden, marking=self.node_genes[-2].marking + 1
                )
                self.node_genes.append(new_node)
                self.node_genes.sort(key=lambda node: node.marking)
                self.connect_genes.append(
                    ConnectGene(
                        connection.in_node_marking,
                        new_node.marking,
                        np.random.normal(0, 1),
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
                connection.disabled = True

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
        i, j = 0, 0

        for marking in sorted(unique_markings):
            if (
                marking == self.connect_genes[i].marking
                and marking == other.connect_genes[j].marking
            ):
                weight_diff.append(
                    abs(self.connect_genes[i].weight - other.connect_genes[j].weight)
                )
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

        excess = (
            len(self.connect_genes) - i + len(other.connect_genes) - j
        )
        N = (
            1
            if len(unique_markings) < 20
            else len(self.connect_genes)
            if len(self.connect_genes) > len(other.connect_genes)
            else len(other.connect_genes)
        )
        return (
            (c1 * excess) / N
            + (c2 * disjoint) / N
            + c3 * sum(weight_diff) / len(weight_diff)
        )

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
        G = nx.DiGraph()
        G.add_nodes_from([node.marking for node in child_nodes])
        G.add_edges_from(
            [
                (connect.in_node_marking, connect.out_node_marking)
                for connect in child_connect_genes
            ]
        )
        if sum([1 for _ in nx.simple_cycles(G)]) == 0:
            return Individuum(child_nodes, child_connect_genes)
        else:
            return None

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
        # BIAS SUMMATION
        for connection in [
            connect
            for connect in self.connect_genes
            if connect.in_node_marking in [node.marking for node in self.bias_nodes]
            and not connect.disabled
        ]:
            out_node = [
                node
                for node in self.node_genes
                if node.marking == connection.out_node_marking
            ][0]
            out_node.sum += connection.weight
            out_node.acts += 1
        while next_layer:
            current_layer = next_layer
            not_activated = []
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
                else:
                    not_activated.append(node)
            for node in not_activated:
                current_layer.remove(node)

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
