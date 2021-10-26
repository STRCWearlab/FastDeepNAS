from typing import List

import networkx as nx
import matplotlib.pyplot as plt

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from NSC import NSC
    from model_generator import ModelSpec
    from nas_utils import uniquify
else:
    # uses current package visibility
    from .NSC import NSC
    from .model_generator import ModelSpec
    from .nas_utils import uniquify


class DNNGraph:
    """Class describing the representation of a DNN in terms of a directed acyclic graph.

    Arguments:
    model -- the NSC representation of the DNN
    """

    def __init__(self, model):

        self.model = model

        self.node_names = self.name_nodes()

        self.graph = self.render_network()

    def render_network(self):
        """Render the DAG using networkx/pygraphviz.

        Generates a networkx representation of the DNN, draws it using matplotlib, converts it to pydot format, colours
        and names each node, and returns a pydot graph."""

        graph = nx.DiGraph()
        pairs = []

        for k, pred in enumerate(self.model.edges.T):
            for i, connection in enumerate(pred):
                if connection == 1:
                    pairs.append([self.node_names[k], self.node_names[i]])

        if len(self.model.loose_ends) > 1:
            for end in self.model.loose_ends:
                pairs.append([self.node_names[end], self.node_names[-1]])
        else:
             self.node_names = self.node_names[:-1]
        # Add final Linear classification layer
        self.node_names.append('Linear')
        pairs.append([self.node_names[-2], 'Linear'])

        graph.add_nodes_from(self.node_names)
        graph.add_edges_from(pairs)

        plt.plot(1)

        nx.draw(graph, with_labels=True)

        plt.show()

        # Convert graph to pydot

        graph = nx.drawing.nx_pydot.to_pydot(graph)

        # Format graph
        for node in graph.get_node_list():
            node_name = node.get_name()

            if 'Cat' in node_name:
                node.set_shape('box')
                node.set_style('filled')
                node.set_fillcolor('coral')
            if 'Add' in node_name:
                node.set_shape('box')
                node.set_style('filled')
                node.set_fillcolor('lightcoral')
            if 'Avg. Pool' in node_name:
                node.set_style('filled')
                node.set_fillcolor('steelblue')
            if 'Max Pool' in node_name:
                node.set_style('filled')
                node.set_fillcolor('cornflowerblue')
            if 'Conv' in node_name:
                node.set_style('filled')
                node.set_fillcolor('darkorchid')
            if 'ID' in node_name:
                node.set_style('filled')
                node.set_color('lightcyan')
            if 'In' in node_name:
                node.set_shape('box')
            if 'Linear' in node_name:
                node.set_shape('box')

        return graph

    def name_nodes(self):
        """Give each node in the DAG a unique name which contains all identifying information.

        The name specifies the node index and operation performed, as well as relevant information including
        the size and number of kernels in a conv layer, and the dimensionality reduction factor in a pooling layer.
        Returns a list of names (str).
        """

        shortnames = {'Input': 'In',
                      'Convolution': 'Conv',
                      'Max Pooling': 'Max Pool',
                      'Average Pooling': 'Avg. Pool',
                      'Identity': 'ID',
                      'Elementwise Addition': 'Add',
                      'Concatenation': 'Cat',
                      'Terminal': 'Cat'}

        labels = ['{} - {}'.format(layer[0], shortnames[layer[1].Type]) for layer in self.model.encoding.iterrows()]

        kernel_sizes = [layer[1].Kernel_size for layer in self.model.encoding.iterrows()]
        kernel_numbers = [layer[1].N_kernels for layer in self.model.encoding.iterrows()]
        for i, size in enumerate(kernel_sizes):
            if size != 0 and kernel_numbers[i] == 0:
                labels[i] += ' ({0})'.format(size)
            elif size != 0 and kernel_numbers[i] != 0:
                labels[i] += ' ({} x {})'.format(size, kernel_numbers[i])

        return labels
