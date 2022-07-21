#
# Connections between python's Probabilistic Graphical Model (pgm)
# library pgmpy and EfProb, in particular with respect to Bayesian
# network models and inference
#
# Copyright: Bart Jacobs; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2018-04-24
#
from efprob_dc import *
from baynet import *

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#from pgmpy.inference import BeliefPropagation

import pydot
from PIL import Image

import sys

#####################################################################
#
# The main functions in this file are:
#
# - factorise: it turns an EfProb joint state together with a pydot
#   graph into a pmgpy Bayesian model
#
# - stretch: it linearises a pgmpy Bayesian model to a list of
#   composable EfProb channels, where the first entry is a state
#
# - inference_query: it computes, starting from the output of the
#   stretch function, the marginal distribution, given a list of
#   evidence.
#
#####################################################################



#####################################################################
#
# Auxiliary pydot graph functions
#
#####################################################################

#
# Save graph as image in the directory where the function is called
# and display it. When no name is provided, the file name is graph.png
#
# In case of errors, do: pip3 install --upgrade pillow
#
# Kill all images with: pkill display
#
def graph_image(graph, name=None):
    if name == None:
        name = "graph"
    working_directory = os.getcwd()
    file_path = working_directory + "/" + name + ".png"
    graph.write_png(file_path) 
    f = open(file_path, "rb")
    Image.open(f).show()

#
# Return list of parents of a node in a graph; this list contains an
# arbitrary order, and may be different if the function is run twice
# on the same node, since the edges of a node form a ***set***, not a
# ***list***.
#
def get_parents(node):
    return [e.get_source() for \
            e in node.get_parent_graph().get_edges() if \
            e.get_destination() == node.get_name() ]

#
# Return the list of children of a node, in arbitrary order.
#
def get_children(node):
    return [e.get_destination() for \
            e in node.get_parent_graph().get_edges() if \
            e.get_source() == node.get_name() ]


#
# Swap the entries in a joint state (in EfProb), using a permutation
# given by a list of positions.
#
def domain_swaps(state, swap_list):
    l = len(swap_list)
    doms = state.dom
    if l != len(doms):
        raise Exception('Swapping requires equal lenght inputs')
    # form the new domain, first as list of domains, then as joint domain
    swapped_doms = []
    for i in range(l):
        swapped_doms.append(doms.get_nameditem(swap_list[i]))
    swapped_doms = reduce(lambda d1, d2: d1 + d2, swapped_doms)
    # form the channel that does the swapping
    swap_chan = chan_fromklmap(lambda *xs: 
                               point_state(tuple([xs[swap_list[i]] 
                                                  for i in range(l)]),
                                           swapped_doms),
                               doms, swapped_doms)
    return swap_chan >> state

#
# Reorder the domains of an EfProb state, in accordance with the
# provided list of domains 'domain_names'
#
def reorder_state_domains(state, domain_names):
    ls = len(state.dom)
    ld = len(domain_names)
    state_names = [state.dom.names[i].name for i in range(ls)]
    if ld != ls or set(domain_names) != set(state_names):
        raise Exception('Non-matching domains in domain-reordering')
    swaps = []
    for i in range(ld):
        # find domain_names[i] in state_names, and extend the swap list
        # with its position
        for j in range(ld):
            if state_names[j] == domain_names[i]:
                swaps.append(j)
                break
    return domain_swaps(state, swaps)


#
# Pick randomly n different items from a list
#
def pick_from_list(items, n):
    l = len(items)
    if n > l:
        raise Exception('List is too short to pick so many items')
    # make deep copy, in order not to affect the list of items
    copy = [a for a in items]
    picks = []
    for i in range(n):
        r = random.randint(0, l-i-1)
        a = copy[r]
        copy.remove(a)
        picks.append(a)
    return picks


#####################################################################
#
# Auxiliary functions, from pgmpy to efprob (or pydot)
#
#####################################################################

#
# Turn a name (string) and cardinality (number) into a named EfProb
# domain, with entiries of the form name_i
#
def efprob_domain(name, card):
    items = [ name + "_" + str(i) for i in range(card)]
    return Dom(items, names = Name(name))
    
#
# Turn the nodes of a graphical model into a list of EfProb domains
#
def efprob_domains_of_pgm(model):
    nodes = model.nodes()
    domains = {}
    for n in nodes:
        domains[n] = efprob_domain(n, model.get_cardinality(n))
    return domains

#
# The incoming edges of a node in a graphical model are turned into a
# EfProb channel. This is done for each node, and yields a dictionary
# with a channel for each node (as key)
#
def efprob_channels_of_pgm(model):
    nodes = model.nodes()
    cpts = {}
    for n in nodes:
        cpd = model.get_cpds(node=n)
        card_n = cpd.variable_card
        # first extract codomain
        cod = efprob_domain(cpd.variable, card_n)
        reverse_parent_nodes = cpd.get_evidence()
        parent_nodes = [reverse_parent_nodes[i] 
                        for i in range(len(reverse_parent_nodes)-1, -1, -1)]
        parents_num = len(parent_nodes)
        #print("* Handling node:", n, card_n, parent_nodes )
        vals = cpd.get_values() 
        if parents_num == 0:
            # include a state, not a channel, since there is no domain
            cpts[n] = State(vals.transpose()[0], 
                            efprob_domain(n, 
                                          model.get_cardinality(n))).as_chan()
        else:
            # extract a channel; first find its domain
            doms = [efprob_domain(pn, model.get_cardinality(pn)) 
                    for pn in parent_nodes]
            dom = reduce(lambda d1, d2: d1 + d2, doms)
            #print("Domain:", len(dom), dom)
            states = []
            steps = 1
            for i in range(parents_num):
                parent_i = parent_nodes[i]
                card_i = model.get_cardinality(parent_i)
                steps *= card_i
            states = [State(vals.transpose()[j], cod) for j in range(steps)]
            cpts[n] = chan_from_states(states, dom)
    return cpts

#
# Extra a pydot graph of a graphical model
#
def pydot_graph_of_pgm(model):
    graph = pydot.Dot(graph_type='digraph')
    nodes = model.nodes()
    for n in nodes:
        graph.add_node(pydot.Node(n))
    edges = model.edges()
    for e in edges:
        graph.add_edge(pydot.Edge(e))
    return graph


#####################################################################


#
# Turn an EfProb state and a pydot graph into pmgpy Bayesian model via
# disintegration, following the graph structure. The resulting model
# has the same underlying graph.
#
# Assumptions:
# - all domains in the state have a name; all these names are unique
# - the set of these names from the state is the same as the set of 
#   names in the graph
#
# This function requires more testing, esp. wrt. how cpd_array is formed
#
def factorise(state, graph):
    state_names = [n.name for n in state.dom.names]
    graph_nodes = graph.get_nodes()
    graph_names = [n.get_name() for n in graph_nodes]
    l = len(state_names)
    if l != len(graph_names):
        raise Exception('Missing domain names of state in factorisation')
    if set(state_names) != set(graph_names):
        raise Exception('Non-matching graph and state names in factorisation')
    # make dictionary of names and corresponding masks
    masks = {}
    for i in range(l):
        ls = l * [0]
        ls[i] = 1
        masks[state_names[i]] = ls
    # dictionary to be filled with cpts = conditional probability tables
    model = BayesianModel()
    model.add_nodes_from(graph_names)
    for node in graph_nodes:
        parents = get_parents(node)
        key = node.get_name()
        mask_cod = masks[key]
        if len(parents) == 0:
            # marginalise for initial nodes
            initial_state = state % mask_cod
            if len(initial_state.dom) > 1:
                raise Exception('Initial states must have dimension 1')
            dom_card = len(initial_state.dom[0])
            state_array = initial_state.array
            cpd_array = np.zeros((dom_card,1))
            for i in range(dom_card):
                cpd_array[i][0] = state_array[i]
            cpd = TabularCPD(variable = key, 
                             variable_card = dom_card,
                             values = cpd_array)
            model.add_cpds(cpd)
        else:
            # add edges and form conditional probility for internal nodes
            for p in parents:
                model.add_edge(p, key)
            mask_dom = mask_summation([masks[p] for p in parents])
            chan = state[ mask_cod : mask_dom ]
            cod = chan.cod
            dom = chan.dom
            if len(cod) > 1:
                raise Exception('Domains must have dimension 1')
            chan_array = chan.array
            print("* ", key, len(dom[0]), len(cod[0]), chan_array.shape )
            prod = reduce(operator.mul, [len(d) for d in dom], 1)
            cpd_array = np.zeros((len(cod[0]), prod))
            for i in range(len(dom[0])): 
                cpd_array[i] = [ chan_array[i][j] for 
                                 j in np.ndindex(*chan_array[i].shape) ]
            cpd = TabularCPD(variable = key, 
                             variable_card = len(cod[0]), 
                             values = cpd_array,
                             evidence = parents,
                             evidence_card = [len(d) for d in dom])
            model.add_cpds(cpd)
    if not model.check_model():
        raise Exception('Constructed model does not pass check')
    return model


#####################################################################

#
# Turn a Probabilistic Graphical Model (pgm, as in pgmpy) into a
# sequence of channels, together with a node_pointer dictionary which
# tells for each node in the model where it occurs in this sequence.
#
# Optionally, via additional True/False settings
#
# - a pydot graph is produced of the pgm (with additional copy and
#   identity nodes), indicating the burocratic channels for copying
#   and reordering
#
# - an 'observed' version, both of the sequence of channels and of the
#   graph, is produced, in which internal nodes become 'observable'
#   (or accessible). This version is useful for producing the joint
#   distribution from the model.
#
def stretch(pgm, graph_output=False, observed=False, silent=True,
            stretch_tries=1000):
    #
    # Extraction of relevant data from input
    #
    nodes = pgm.nodes
    nodes_num = len(nodes)
    efprob_domains = efprob_domains_of_pgm(pgm)
    channels = efprob_channels_of_pgm(pgm)
    # collect parents and children in a dictionary and initial nodes in a list
    children = {}
    parents = {}
    for n in nodes:
        children[n] = []
        parents[n] = []
    for e in pgm.edges():
        children[e[0]].append(e[1])
        parents[e[1]].append(e[0])
    # collect number of elements of each node's domain in a dictionary
    node_sizes = {}
    for ck in channels.keys():
        node_sizes[ck] = len(channels[ck].cod[0])
    def list_size(node_list):
        return reduce(operator.mul, [node_sizes[n] for n in node_list], 1)
    initial_nodes = []
    final_nodes = []
    # non-initial and non-final nodes
    intermediate_nodes = []
    for n in nodes:
        if len(parents[n]) == 0:
            initial_nodes.append(n)
        else:
            if len(children[n]) == 0:
                final_nodes.append(n)
            else:
                intermediate_nodes.append(n)
            # update with the order as actually used in the channel
            parents[n] = [dn.name for dn in channels[n].dom.names]
    #
    # Initialisation of data structures that will be build up: (a)
    # list of channels, (b) nodes pointers, (c) node copies, (d)
    # graph, optionally
    #
    # (a) List of channels, where the first entry consists of the
    # (product of the) initial states. This list of channels is the
    # main part of the output
    #
    channel_list = [ idn(Dom([], names=Name("Empty"))) ]
    pointer = 1
    # 
    # (b) Dictionary with index level (natural number) for each node
    #
    node_pointer = {}
    #
    # (c) Dictionary for keeping track of how many copies are needed of a
    # node, so that copying can be done lazily, whenever needed.
    #
    node_copies = {}
    #
    # (d) Optional graph of the linearised graph; original nodes are
    # in green
    #
    if graph_output:
        stretched_graph = pydot.Dot(graph_type='digraph')
        # auxiliary function for addition of a named black bullet to
        # the graph, to be used as copy node; set fontsize to 12 to
        # see the names of the copy nodes, for debugging. Otherwise
        # the name is suppressed
        def add_copy_node(name):
            stretched_graph.add_node(
                pydot.Node(name,
                           width=0.15,
                           height=0.15,
                           fixedsize=True,
                           style="filled", 
                           fillcolor="black",
                           fontsize=0))
    #
    # Phase one: find the most "narrow" way of stretching the Bayesian
    # network into linear form, via repeated (stretch_tries many
    # times) random trials. The narrowness of a stretching is
    # calculated as the size of the states involved, expressed as
    # product of the number of items of component states. There are
    # probably more systematic ways of finding a narrow stretching.
    #
    stretch_size = sys.maxsize
    stretch_outcome = []
    processed_initial_nodes = set([])
    for st in range(stretch_tries):
        current_size = 0
        current_stretch = []
        current_processed_initial_nodes = set([])
        available_nodes = [n for n in initial_nodes]
        unprocessed_nodes = []
        node_copies = {}
        for n in initial_nodes:
            node_copies[n] = len(children[n]) + (1 if observed else 0)
        unprocessed_nodes = []
        copy_of_intermediate_nodes = [n for n in intermediate_nodes]
        # find a random permutation of the intermediate nodes in the
        # list 'unprocessed_nodes'; the code below will go through
        # this list in order to find a stretching
        #
        # To be done, using instead: r = np.random.permutation(n)
        # unprocessed_nodes = unprocessed_nodes[r]
        for i in range(len(intermediate_nodes)):
            rand_index = random.randint(0, len(intermediate_nodes) - i - 1)
            rand_node = copy_of_intermediate_nodes[rand_index]
            unprocessed_nodes.append(rand_node)
            copy_of_intermediate_nodes.remove(rand_node)
        # put final nodes first, so that they are handled first,
        # keeping the size down
        unprocessed_nodes = final_nodes + unprocessed_nodes
        iterations = 0
        while len(unprocessed_nodes) > 0:
            for un in unprocessed_nodes:
                iterations += 1
                parents_un = parents[un]
                proceed = all([pn in available_nodes for pn in parents_un])
                if not proceed:
                    # handle this node un later
                    continue
                #print("==> Parents found for node", un)
                current_stretch.append(un)
                for pn in parents_un:
                    node_copies[pn] -= 1
                    if node_copies[pn] == 0:
                        available_nodes.remove(pn)
                    if pn in initial_nodes:
                        current_processed_initial_nodes.add(pn)
                available_nodes.append(un)
                node_copies[un] = len(children[un]) + (1 if observed else 0)
                s = list_size(available_nodes)
                if s > current_size:
                    current_size = s
                #print("Available nodes:", s)
                unprocessed_nodes.remove(un)
        # print("Stretch attempt", st, "ends with size:", current_size)  
        if current_size < stretch_size:
            stretch_size = current_size
            stretch_outcome = current_stretch
            processed_initial_nodes = current_processed_initial_nodes
    for n in initial_nodes:
         if not n in processed_initial_nodes:
             #print("Post addition of", n)
             stretch_outcome.append(n)
    if not silent:
        print("Stretch search done after", stretch_tries, 
              "tries with max size", stretch_size)
    #
    # Phase two: building the actual channel list, based on the list
    # of non-initial nodes 'stretch_outcome'
    #
    available_nodes = [n for n in initial_nodes]
    available_initial_nodes = [n for n in initial_nodes]
    for n in initial_nodes:
        node_copies[n] = len(children[n]) + (1 if observed else 0)
        if graph_output:
            stretched_graph.add_node(pydot.Node(n, 
                                                style="filled", 
                                                fillcolor="green"))
            if len(children[n]) > 0 and observed:
                # n is not a final node
                copy_name = n + "!copy" + str(node_copies[n])
                # add copy nodes
                add_copy_node(copy_name)
                stretched_graph.add_edge(pydot.Edge(n, copy_name))
                stretched_graph.add_edge(
                    pydot.Edge(copy_name, " " + n + " "))
    for node in stretch_outcome:
        if not silent:
            print("==> node", node, "(", pointer, "of", len(stretch_outcome),
                  ") of sizes", [node_sizes[n] for n in available_nodes], 
                  "in total", list_size(available_nodes))
        node_chan = channels[node]
        parents_node = parents[node]
        num_parents_node = len(parents[node])
        copy_list = (len(available_nodes) - len(available_initial_nodes)) * [1]
        #print("Copy list length", len(copy_list))
        new_available_nodes = []
        new_available_initial_nodes = []
        parent_copy_list = []
        parent_chan_list = []
        parent_initial_nodes = []
        copy_position_correction = 0
        for i in range(len(available_nodes)):
            n_i = available_nodes[i]
            initial_n_i = (n_i in available_initial_nodes)
            if n_i in parents_node:
                node_copies[n_i] -= 1
                if initial_n_i:
                    if not silent:
                        print("--> initial node", n_i, "added")
                    copy_position_correction += 1
                    parent_initial_nodes.append(n_i)
                    parent_chan_list.append(channels[n_i])
                    # pointer - 1 is used since the initial node is
                    # added to the previous channel
                    node_pointer[n_i] = pointer - 1
                    if node_copies[n_i] > 0:
                        parent_copy_list.append(2)
                        parent_initial_nodes.append(n_i)
                    else:
                        parent_copy_list.append(1)
                else:
                    new_available_nodes.append(n_i)
                    if node_copies[n_i] > 0:
                        # more copies of un needed later on
                        new_available_nodes.append(n_i)
                        copy_list[i - copy_position_correction] += 1
            else:
                if initial_n_i:
                    copy_position_correction += 1
                    new_available_initial_nodes.append(n_i)
                else:
                    new_available_nodes.append(n_i)
        available_nodes = parent_initial_nodes + new_available_nodes
        available_initial_nodes = new_available_initial_nodes
        copy_list = parent_copy_list + copy_list
        # update the last channel with the required copying
        lcs = len(channel_list)
        last_channel = channel_list[lcs-1]
        if len(parent_initial_nodes) > 0:
            parent_chan = reduce(lambda c1, c2: c1 @ c2, parent_chan_list)
            last_channel = parent_chan @ last_channel
        channel_list[lcs-1] = copy_chan(last_channel, copy_list)
        #print(" ", available_nodes, channel_list[lcs-1].cod.names )
        #
        # Now search for the precise positions of the parent nodes
        #
        #print("==> Searching locations of parents", parents_node, 
        #      "with availables", available_nodes)
        search_copy_of_nodes = [u for u in available_nodes]
        swaps = list(range(len(available_nodes)))
        #
        # find the actual occurrences of un's parents in available domains
        #
        for i in range(num_parents_node):
            # print("... searching for parent: ", parents_node[i] )
            #
            # try to locate i-th parent among domains
            #
            for j in range(len(available_nodes)):
                if search_copy_of_nodes[j] == parents_node[i]:
                    #print("Parent", parents_node[i], "found at", j)
                    tmp = swaps[j]
                    swaps[j] = swaps[i]
                    swaps[i] = tmp
                    search_copy_of_nodes[j] = search_copy_of_nodes[i]
                    search_copy_of_nodes[i] = parents_node[i]
                    break
        #
        # incorporate swaps into available nodes and arguments
        #
        inv_swaps = len(available_nodes) * [0]
        swapped_doms = []
        swapped_available_nodes = []
        for i in range(len(available_nodes)):
            inv_swaps[swaps[i]] = i
            swapped_available_nodes.append(available_nodes[swaps[i]])
        available_nodes = swapped_available_nodes
        #print( swaps, inv_swaps )
        #print("Swapped nodes", available_nodes)
        swapped_doms = [efprob_domains[n] for n in swapped_available_nodes]
        swapped_dom = reduce(lambda d1, d2: d1 + d2, swapped_doms)
        #
        # Build the channel that incorporates the node and does the swapping
        #
        node_chan_id = node_chan
        diff = len(available_nodes) - num_parents_node
        if diff > 0:
            identities_doms = []
            for i in range(diff):
                d_i = swapped_dom.get_nameditem(i + num_parents_node)
                identities_doms.append(d_i)
            identities_dom = reduce(lambda d1, d2: d1 + d2, identities_doms)
            identities = idn(identities_dom)
            node_chan_id = node_chan @ identities
        #
        # Add the channel to the list, with its domains permuted
        # 
        channel_list.append(perm_chan(node_chan_id, dom_perm = inv_swaps))
        #print("Swapped chan dom", channel_list[len(channel_list)-1].dom.names )
        node_pointer[node] = pointer
        pointer += 1
        node_copies[node] = len(children[node]) + (1 if observed else 0)
        if node in initial_nodes:
            available_initial_nodes.remove(node)
        available_nodes = available_initial_nodes + \
                          [node] + available_nodes[num_parents_node:]
        #print("Nodes at end", available_nodes)
        #print("Codomain at end:", channel_list[len(channel_list)-1].cod.names )
        #
        # Add node to the graph, with links to its parents, if needed
        # via (binary) copying.
        #
        if graph_output:
            stretched_graph.add_node(
                pydot.Node(node, style="filled", fillcolor="green"))
            for i in range(num_parents_node):
                parent_node_i = parents_node[i]
                copies_i = node_copies[parent_node_i]
                copy_name_i = parent_node_i + "!copy" + str(copies_i)
                copy_name_Si = parent_node_i + "!copy" + str(copies_i + 1)
                parent_node_i_children_num = len(children[parent_node_i])
                if not observed:
                    if parent_node_i_children_num == 1:
                        # parent_node_i --> node is the only edge
                        # parent_node_i --> ...
                        stretched_graph.add_edge(
                            pydot.Edge(parent_node_i, node))
                    else:
                        if parent_node_i_children_num == copies_i + 1:
                            # add just one copy and connect its base to parent_i
                            add_copy_node(copy_name_i)
                            stretched_graph.add_edge(
                                pydot.Edge(parent_node_i, copy_name_i))
                            stretched_graph.add_edge(
                                pydot.Edge(copy_name_i, node))
                        else:
                            if copies_i == 0:
                                # connect directly to last copy node
                                stretched_graph.add_edge(pydot.Edge(
                                    copy_name_Si, node))
                            else:
                                # add copy node and connect to previous
                                add_copy_node(copy_name_i)
                                stretched_graph.add_edge(
                                    pydot.Edge(copy_name_Si, copy_name_i))
                                stretched_graph.add_edge(
                                    pydot.Edge(copy_name_i, node))
                else:
                    # observed case
                    if copies_i == 1:
                        # connect directly to last copy node
                        # print("no more copying needed")
                        stretched_graph.add_edge(pydot.Edge(
                            copy_name_Si, node))
                    else:
                        # add copy node and connect to previous
                        add_copy_node(copy_name_i)
                        stretched_graph.add_edge(
                            pydot.Edge(copy_name_Si, copy_name_i))
                        stretched_graph.add_edge(
                            pydot.Edge(copy_name_i, node))
            if observed and node_copies[node] > 1:
                # add copy node for children to connect to in next round
                copy_name_i = node + "!copy" + str(node_copies[node])
                add_copy_node(copy_name_i)
                stretched_graph.add_edge(pydot.Edge(node, copy_name_i))
                stretched_graph.add_edge(
                    pydot.Edge(copy_name_i, " " + node + " "))
    #
    # Collect the results in a dictionary
    # 
    result = {
        'pointer'  : node_pointer,
        'channels' : channel_list
    }
    if graph_output:
        result['graph'] = stretched_graph
    return result

#
# Successively apply state transformation >> to the initial state, by
# applying the successive channels one-by-one. The chan-list input is
# typically produced by the stretch function.
#
def evaluate_stretch(chan_list):
    state = chan_list[0] >> init_state
    for i in range(len(chan_list)-1):
        state = chan_list[i+1] >> state
    return state

#
# Return a list of (first) marginals from a channel-list and a pointer
# dictionary, as produced by the stretch function. 
#
def marginals_stretch(stretch_dict):
    chan_list = stretch_dict['channels']
    state_list = [chan_list[0]]
    for i in range(len(chan_list) - 1):
        state_list.append(chan_list[i+1] >> state_list[i])
    ptr = stretch_dict['pointer']
    result = []
    for n in ptr.keys():
        if ptr[n] > 0:
            # print(n, ptr[n], len(state_list) )
            state = state_list[ptr[n]-1]
            mask = len(state.dom) * [0]
            mask[0] = 1
            result.append(state % mask)
    return result


#
# This function operates on the output of the 'stretch' function and
# uses it for inference. It takes two additional arguments:
#
# - 'marginal', which is a node in the original model, at which
#   position the probability distribution is computed, given the evidence
#
# - 'evidence_dict', which is a dictionary mapping nodes to list of
#   probabilities in [0,1], acting as predicate values. If the node
#   has cardinality n, the list must have length n.
#
def inference_query(stretch_dict, marginal, evidence_dict = {}):
    chan_list = stretch_dict['channels']
    chan_list_len = len(chan_list)
    state = chan_list[0] >> init_state
    ptr = stretch_dict['pointer']
    ptrkeys = ptr.keys()
    evkeys = evidence_dict.keys()
    if not (marginal in ptrkeys):
        print(marginal, ptrkeys)
        raise Exception('Marginal does not occur')
    if not all([k in ptrkeys for k in evkeys]):
        print(evkeys, ptrkeys)
        raise Exception('Some of the evidence keys do not occur')
    position_marginal = ptr[marginal]
    #print("\nMarginal:", marginal, "at", position_marginal, ", Evidence:", evkeys)
    #
    # The evidence is inserted into a list in parallel to the list of
    # channels, so that updates can happen locally. This evidence-list
    # contains two kinds of entries: 
    #
    # - (0,None) --- nothing will happen here 
    # - (1,pred)
    #
    evidence_list = chan_list_len * [(0,None)]
    for k in evkeys:
        position_k = ptr[k]
        dom = chan_list[position_k].cod
        preds = []
        for i in range(len(dom)):
            if k == dom.names[i].name:
                preds.append(Predicate(evidence_dict[k], dom.get_nameditem(i)))
            else:
                preds.append(truth(dom.get_nameditem(i)))
        weakened_pred = reduce(lambda p1, p2: p1 @ p2, preds)
        evidence_list[position_k] = (1, weakened_pred)
    #
    # Code for inspection
    #
    # for i in range(chan_list_len):
    #     print("\n", i, "pointers", [k for k in ptrkeys if ptr[k] == i])
    #     print("Channel codomain:", len(chan_list[i].cod), chan_list[i].cod.names)
    #     print("Evidence:", "--" if evidence_list[i][0] == 0 
    #           else str([k for k in evkeys if ptr[k] == i]) + \
    #           str(evidence_list[i][1].dom.names))
    #     if i + 1 < chan_list_len:
    #         print(chan_list[i].cod.names)
    #         print(chan_list[i+1].dom.names)
    #
    # Start with forward state transformation up to the marginal
    # position, updating with predicates, if any, along the way
    #
    for i in range(position_marginal):
        if evidence_list[i][0] == 0:
            # no predicate at this point, transform existing state
            state = chan_list[i+1] >> state
        else:
            # update the current state with the predicate first, then
            # transform, i.e. do forward inference
            state = state / evidence_list[i][1]
            state = chan_list[i+1] >> state
    #print("\nDomain of state:", state.dom.names)
    #
    # Continue with backward predicate transformation, adding
    # predicates, if any, along the way, via conjunction &. This
    # addition only starts when a predicate is found.
    #
    hit = False
    for i in range(chan_list_len - position_marginal - 1):
        pos_i = chan_list_len - i - 1
        if evidence_list[pos_i][0] == 0:
            if hit:
                pred = chan_list[pos_i] << pred
        else:
            if not hit:
                hit = True
                pred = chan_list[pos_i] << evidence_list[pos_i][1]
            else:
                # take the conjunction of the current predicate with the
                # predicate at this point, then transform
                pred = pred & evidence_list[pos_i][1]
                pred = chan_list[pos_i] << pred
    mask = len(state.dom) * [0]
    for i in range(len(state.dom)):
        if state.dom.names[i].name == marginal:
            mask[i] = 1
            break
    if not hit:
        return state % mask
    else:
        state = state / pred
        return state % mask
    #print("\nDomain of predicate:", pred.dom.names)
    #
    # Combine the results of the forward and backward operations by
    # updating the state once more, and taking the appropriate (first)
    # marginal.
    #


#
# Remove parts of a Bayesian model so that it still contains
# selected_nodes; a node can be removed from pgm if non of its
# descendants is in selected_nodes.
#
def clip_model(bay_mod, selected_nodes):
    pgm = bay_mod.copy()
    nodes = pgm.nodes
    # collect children in a dictionary
    children = {}
    for n in nodes:
        children[n] = []
    for e in pgm.edges():
        children[e[0]].append(e[1])
    # collect descendants iteratively
    unvisited_nodes = [n for n in nodes]
    descendants = {}
    while len(unvisited_nodes) > 0:
        for n in unvisited_nodes: 
            if any([c in unvisited_nodes for c in children[n]]):
                continue
            descendants[n] = [n] + children[n] + [m for c in children[n] 
                                                for m in descendants[c] ]
            unvisited_nodes.remove(n)
    # remove duplicates
    for n in nodes:
        descendants[n] = set(descendants[n])
    nodes_copy = [n for n in nodes]
    for n in nodes_copy:
        if any([(on in descendants[n]) for on in selected_nodes]):
            continue
        pgm.remove_node(n)
    return pgm

def stretch_and_infer(pgm, marginal, evidence, silent=True):
    #
    # Extraction of relevant data from input
    #
    selected_nodes = [marginal] + list(evidence.keys())
    clipped_pgm = clip_model(pgm, selected_nodes)
    if not silent:
        print("Observed:", selected_nodes)
        graph = pydot_graph_of_pgm(clipped_pgm)
        graph_image(graph, "test")
    stretched_pgm = stretch(clipped_pgm, silent=silent)
    return inference_query(stretched_pgm, marginal, evidence)        






