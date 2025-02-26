import networkx as nx
import matplotlib.pyplot as plt
import random as ra
import ast


def is_int(n):
    try:
        float_n = float(n)
        int_n = int(float_n)
    except ValueError:
        return False
    else:
        return float_n == int_n


def is_float(n):
    try:
        float_n = float(n)
    except ValueError:
        return False
    else:
        return True


def convert_to_numeric(num):
    # for num in string_list:
    if is_int(num):
        #print(num, 'can be safely converted to an integer.')
        return int(num)
    elif is_float(num):
        #print(num, 'is a float with non-zero digit(s) in the fractional-part.')
        return float(num)


class SAIGraph:

    def __init__(self, args):
        self._args=args
        # if self._args.edgelist_filename and self._args.graph_type:
        #     print(
        #         "SAIGraph Warning: too many config options. We will use the FILENAME\n")

        ra.seed(args.seed)

        if self._args.graph_from == "file" and self._args.edgelist_filename:
            # # read edgelist and sort it
            # #read lines of text file and split into words
            # lines = [line.split() for line in open(self._args.edgelist_filename, "r")]
            # #sort lines for different columns, numbers converted into integers to prevent lexicographical sorting
            # lines.sort(key = lambda x: (int(x[0]), int(x[1]), float(x[2])))
            # # print(lines)
            # edgelist = [' '.join(l) for l in lines]
            # print(edgelist)
            # load graph from edge list, then sort the nodes before storing it in the sai_graph
            tmp_g = nx.read_edgelist(self._args.edgelist_filename, nodetype=int, data=(("weight", float),)) # commented, now we fist sort the file to avoid problems
            self.sai_graph = nx.Graph()
            self.sai_graph.add_nodes_from(sorted(tmp_g.nodes(data=True)))
            self.sai_graph.add_edges_from(tmp_g.edges(data=True))
            if self.sai_graph.get_edge_data(0,1) == {}:
                for e in self.sai_graph.edges():
                    self.sai_graph[e[0]][e[1]]['weight'] = 1
            # self.sai_graph = nx.parse_edgelist(edgelist, nodetype=int, data=(("weight", float),))
            print("SAIGraph: loaded graph from file ", self._args.edgelist_filename)

            for n in self.sai_graph.nodes():
                print(f'{n}:{list(self.sai_graph.neighbors(n))}')
            print(self.sai_graph.edges(data=True))

        elif self._args.graph_from == "synth":
            # generate synthetic graph
            if self._args.graph_synth_type == "stochastic_block_model":
                ga = ast.literal_eval(self._args.graph_synth_args)
                self.sai_graph = nx.__getattribute__(self._args.graph_synth_type)(**ga)
            else:
                ga = map(convert_to_numeric, self._args.graph_synth_args.split(','))
                self.sai_graph = nx.__getattribute__(self._args.graph_synth_type)(*ga)

            print("SAIGraph: generated synthetic graph of type",
                  self._args.graph_synth_type, "with parameters", self._args.graph_synth_args)

            # TODO: there could be different ways to assign weights to edges. For the moment, we set all weights to 1
            # adding weights to edges
            for e in self.sai_graph.edges():
                self.sai_graph[e[0]][e[1]]['weight'] = 1
                #if args.pablo:
                    # self.sai_graph[e[0]][e[1]]['weight'] = 1
                # else:
                #     # FIXME: this should be removed, it adds undesired randomness
                #     self.sai_graph[e[0]][e[1]]['weight'] = ra.uniform(0.01, 3)
                
        elif self._args.graph_from == "preset":
            # load graph from networkx dataset
            self.sai_graph = nx.__getattribute__(self._args.graph_preset)()
            # forcing weights to edges
            for e in self.sai_graph.edges():
                # self.sai_graph[e[0]][e[1]]['weight'] = ra.uniform(0.01, 3)
                self.sai_graph[e[0]][e[1]]['weight'] = 1
        else:
            print("SAIGraph Error: no parameter provided, can't create the graph\n")
            exit(1)

        # plot graph
        weights = nx.get_edge_attributes(self.sai_graph, 'weight').values()

        filename = "stats/" + args.outfolder + "/graph.pdf"
        f = plt.figure()
        nx.draw(self.sai_graph, pos=nx.spring_layout(self.sai_graph),
                with_labels=True,
                width=list(weights))
        f.savefig(filename)

        # export graph
        filename = "stats/" + args.outfolder + "/graph.edgelist"
        nx.write_weighted_edgelist(self.sai_graph, filename)
    
    def get_nodes_with_communities(self):
        return dict()
