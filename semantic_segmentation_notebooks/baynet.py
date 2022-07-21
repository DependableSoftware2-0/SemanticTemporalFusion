#
# Bayesian network library, prototype version
#
# Copyright: Bart Jacobs; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2018-04-24
#
import pydot
from PIL import Image
import os
from efprob_dc import *

#####################################################################
#
# This file contains three main functions, namely:
#
# factorise: turn a (matching) joint state and a graph into a Bayesian
# network.
#
# flatten: turn a (matching) graph and Bayesian network into a joint
# state.
#
# state_graph_match: compute the distance between a joint state and
# its reconstructed version according to a given graph.
#
#####################################################################


#
# Auxiliary graph functions
#

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
# Turn a state and a graph into a dictionary of conditional
# probability tables (cpts), consisting of initial states and channels
# for each of the edges in the graph.
#
# Assumptions:
# - all domains in the state have a name; all these names are unique
# - the set of these names from the state is the same as the set of 
#   names in the graph
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
    cpts = {}
    for node in graph_nodes:
        parents = get_parents(node)
        key = node.get_name()
        mask_cod = masks[key]
        if len(parents) == 0:
            # marginalise for initial nodes
            cpts[key] = state % mask_cod
        else:
            # form conditional probility for internal nodes
            mask_dom = mask_summation([masks[p] for p in parents])
            cpts[key] = state[ mask_cod : mask_dom ]
    return cpts


#
# Two more auxiliary functions
#

#
# Swap the entries in a joint state, using a permutation given by a
# list of positions.
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
# 
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
# A graph together with a dictionary of "conditional probability
# tables" (cpts) is turned into a joint state over all the nodes of
# the graph. The entries in the cpts are states and channels whose
# argument numbers and names correspond with the graph. The algorithm
# runs non-deterministically, since it involves iteration over a set.
#
# Assumptions:
#
# - the names of the nodes in the graph coincide with the names of the
#   keys of the cpts dictionary
# - all (co)domains of states and channels in cpts have names, which 
#   coincide with the names in the graph, at the appropriate positions.
#   This latter point is not checked in the code.
#
# The domains in the resulting joint state will be randomly
# ordered. They weill have to be reordered into standard form, if
# needed.
#
def flatten(graph, cpts):
    graph_nodes = graph.get_nodes()
    graph_node_names = [n.get_name() for n in graph_nodes]
    if set(cpts.keys()) != set(graph_node_names):
        print("\nMismatch between:")
        print( cpts.keys() )
        print( graph_node_names )
        raise Exception('Non-matching graph and mapping in flattening')
    #
    # Handle initial nodes from graph, without parents
    #
    initial_nodes = []
    # deep copy
    unprocessed_nodes = graph_nodes[:]
    for n in graph_nodes:
        if len(get_parents(n)) == 0:
            initial_nodes.append(n)
            unprocessed_nodes.remove(n)
    state = reduce(lambda s1, s2: s1 @ s2, [cpts[n.get_name()]
                                            for n in initial_nodes])
    # print("Initial state: ", state )
    #
    # Make copies of the initial states: one more copy than needed by
    # the children so that each initial state remains accessible /
    # observable in the returned state.
    #
    copy_chan = reduce(lambda c1, c2: c1 @ c2, [copy(cpts[n.get_name()].dom, 
                                                     len(get_children(n)) + 1)
                                                for n in initial_nodes])
    state = copy_chan >> state
    #
    # Keep a list of available domains for finding a match of types
    #
    domains = []
    for n in initial_nodes:
        domains += (len(get_children(n)) + 1) * [n.get_name()]
    #
    # Continue with non-initial nodes from graph: cycle through the
    # set of unprocessed nodes until all associated channels have been
    # applied to the current state, and the list of unprocessed nodes
    # is empty.
    #
    iterations = 0
    while len(unprocessed_nodes) > 0:
        for un in unprocessed_nodes:
            iterations += 1
            un_name = un.get_name()
            # print(cpts[un_name])
            parents = [n.name for n in cpts[un_name].dom.names]
            print("* Iteration:", iterations, un_name)
            #print("Parents: ", parents)
            #print("Available nodes to connect to: ", domains)
            swaps = list(range(len(domains)))
            search_copy_of_domains = domains[:]
            # find occurrences of un's parents in domains
            i = 0
            found_all = True
            while i < len(parents):
                #print("Searching for parent: ", parents[i] )
                # try to find i-th parent among domains
                j = 0
                found_i = False
                while j < len(domains):
                    if search_copy_of_domains[j] == parents[i]:
                        found_i = True
                        break
                    j += 1
                if not found_i:
                    # stop handling parent i
                    found_all = False
                    # print("Stop handling parent: ", parents[i])
                    break
                # i-th parent found at j
                #print("==> Parent found! ", i, "=", parents[i], "at", j)
                # swap j |-> i
                swaps[j] = swaps[i]
                swaps[i] = j
                search_copy_of_domains[j] = search_copy_of_domains[i]
                search_copy_of_domains[i] = parents[i]
                # print("Search copy: ", search_copy_of_domains)
                i += 1
            if found_all:
                # all parents found; now update the state with channel of un
                #print("\nAll parents found; domains: ", domains)
                #print( swaps )
                # incorporate swaps
                argument_swaps = list(range(len(swaps)))
                for i in range(len(parents)):
                    tmp = domains[i]
                    domains[i] = domains[swaps[i]]
                    domains[swaps[i]] = tmp
                    tmp = argument_swaps[i]
                    argument_swaps[i] = argument_swaps[swaps[i]]
                    argument_swaps[swaps[i]] = tmp
                    #print(i, swaps[i], domains[i], domains[swaps[i]], domains)
                #print("Swapped domains: ", domains)
                #print("State domain before swap: ", state.dom)
                state = domain_swaps(state, argument_swaps)
                dom = state.dom
                #print("State domain after swap: ", dom)
                un_chan = cpts[un_name]
                identities_dom = []
                for i in range(len(parents), len(domains)):
                    d_i = dom.get_nameditem(i)
                    identities_dom.append(d_i)
                identities_dom = reduce(lambda d1, d2: d1 + d2, identities_dom)
                identities = idn(identities_dom)
                num_children = len(get_children(un))
                if num_children > 0:
                    copies = copy(un_chan.cod, num_children + 1)
                    chan = (copies * un_chan) @ identities
                else:
                    chan = un_chan @ identities
                #print("Channel domain: ", chan.dom)
                # remove domains of channel, and add codomain, several times
                heads = (num_children + 1) * [un_name]
                tails = domains[len(parents):]
                domains = heads + tails
                #print("Domain after update: ", domains)
                state = chan >> state
                # print( state )
                unprocessed_nodes.remove(un)
    #print("Required number of node searches: ", iterations)
    return state


# def KLdivergence(state1, state2):
#     #print("sum", [ state1.array(i) / state2.array(i) for i in state1] )
#     z = zip(list(state1.array), list(state2.array))
#     print("zip ", map(lambda x, y: x * math.log(x/y), z))
#     return -sum(map(lambda x, y: x * math.log(x/y), z))
#     # dom = state1.dom
#     # for d in dom:
#     #     print(d)
#     # print("Dom is ", dom)
#     # return -sum([state1(d) * math.log(state1(d) / state2(d)) for d in dom])


#
# Factorise a joint state and a graph first into a Bayesian network,
# flatten the resulted into a reconstructed joint state, and return
# the distance between the original state and the reconstructed
# one. This distance lies in the unit interval [0,1]. The lower it is,
# the better the state matches the graph wrt. conditional
# independencies. When the distance is 0, the match is perfect.
#
def state_graph_match(state, graph):
    cpts = factorise(state, graph)
    reconstructed_state = flatten(graph, cpts)
    ld = len(state.dom)
    domain_names = [state.dom.names[i].name for i in range(ld)]
    reordered_reconstructed_state = reorder_state_domains(reconstructed_state,
                                                          domain_names)
    # Alternative outputs:
    # - validity reordered_reconstructed_state >= state.as_pred()
    #   But this is not zero (or one) for equal states
    return tvdist(state, reordered_reconstructed_state) 
    # return (tvdist(state, reordered_reconstructed_state), 
    #         KLdivergence(state, reordered_reconstructed_state),
    #         KLdivergence(reordered_reconstructed_state, state))
            
#
# Afterthoughts: here the Bayesian network is derived from the initial
# data (state). One can also "somehow" construct a Bayesian network,
# turn it into a joint state, and compare this to the original data
# (state). Then one asks the question: What is the likelihood of the
# date, given the model (Bayesian network). For this the
# Kullback-Leibner divergence is standardly used (or also
# log-likelihood??). This alternative approach is often used when
# there is limited coverage and/or faith in the data.
