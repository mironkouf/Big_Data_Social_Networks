import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os

# import diko mas #check
from networkx.algorithms.community.centrality import girvan_newman
import itertools

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])
        
    else: 
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]
 
    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions):
    plt.figure(figsize=(10,10))
    
    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G, 
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency 
        width = 0.5                 # edge-width
        )
    plt.show()

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN 
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)
        
        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')
        
        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: 
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            
                else:
                    G,node_names_list = STUDENT_AM_read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            
            else:
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO

                else:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2: 
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else:
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True
            
        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples):

    breakWhileLoop = False

    while not breakWhileLoop:
            print(bcolors.OKGREEN 
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])
            
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC) 
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else: 
                        G = STUDENT_AM_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE

                G = STUDENT_AM_add_hamilton_cycle_to_graph(G,node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else:
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         #ERASE previous node positions

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        graph_layout = 'spring'     # DEFAULT graph layout == spring
                
                    else:
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])

                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:
                        STUDENT_AM_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)

                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:
                        STUDENT_AM_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for 
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        
                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS
                    if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components." + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProvide a number of divisions in { ",NUM_COMPONENTS,",",MAX_NUM_DIVISIONS,"}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        STUDENT_AM_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)

            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                STUDENT_AM_determine_opt_community_structure(G,hierarchy_of_community_tuples)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else:
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        STUDENT_AM_visualize_communities(G,community_tuples,graph_layout,node_positions)

            elif my_option_list[0] == 'E':
                quit()          #EXIT

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

def read_graph_from_csv(NUM_LINKS):

    fb_links = pd.read_csv('fb-pages-food.edges')   # fortothike se ena mhtrwo tupou dataframe
    print("fb_links : ", fb_links)

    fb_links_df = fb_links.iloc[0:NUM_LINKS]        # pairnei apo thn thesh 1-NUM_LINKS+1 mono tou
    print("fb_links_df : ", fb_links_df)

    # svhnei ta duplicates, omws afhnei keno stis theseis autes pou esbhse
    # sbhnoume tis akmes ths morfhs (A,A) giati h methodos edgelist pou xrhsimopoioume parakatw
    # diaxeirizetai tis parallhles akmes alla oxi tous brogxous
    fb_links_loopless_df = fb_links_df.loc[~(fb_links_df["node_1"] == fb_links_df["node_2"])]
    #print(fb_links_loopless_df.to_string())

    # den eimai kai poly sigouros gi auto 
    # ftiaxnoume thn node_names_list 
    node_1_names_list = fb_links_loopless_df["node_1"].tolist()     # pairnoume to 1o collumn
    node_2_names_list = fb_links_loopless_df["node_2"].tolist()     # pairnoume to 2o collumn
    node_names_list = node_1_names_list + node_2_names_list         # merge these 2 lists
    #print(node_names_list)

    # gia na diwxw ta diplotypa apo thn node_names_list
    # Create a dictionary, using the node_names_list items as keys. This will 
    # automatically remove any duplicates because dictionaries cannot have duplicate keys.
    node_names_list = list(dict.fromkeys(node_names_list))
    #print(node_names_list)

    # otan tupwnoume to G metraei tou kombous pou xrhsimopoiei automata mono tou
    # oi akmes pou xrhsimopoiei einai ises me to NUM_LINKS
    G = nx.from_pandas_edgelist(fb_links_loopless_df, "node_1", "node_2", create_using=nx.Graph())
    #print("#################")
    #print(G)
    #print("------------------")

    return G, node_names_list
    '''
    print(bcolors.ENDC + "\t" + '''
        ########################################################################################
        # CREATE GRAPH FROM EDGE_CSV DATA 
        # ...(if needed) load all details for the nodes to the fb_nodes DATAFRAME
        # ...create DATAFRAME edges_df with the first MAX_NUM_LINKS, from the edges dataset
        # ...edges_df has one row per edge, and two columns, named 'node_1' and 'node_2
        # ...CLEANUP: remove each link (from the loaded ones) which is a LOOP
        # ...create node_names_list of NODE IDs that appear as terminal points of the loaded edges
        # ...create graph from the edges_df dataframe
        # ...return the constructed graph
        ########################################################################################
''')

    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(  bcolors.ENDC + "\t\t The routine " 
            + bcolors.OKCYAN + "nx.from_pandas_edgelist(...) "
            + bcolors.ENDC 
            + #creates the graph from a dataframe representing its edges,
               #  one per edge, in two columns representing the tail-node 
                # (node_1) and the head-node (node_2) of the edge.\n)
    '''
######################################################################################################################
# IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM
######################################################################################################################
inside_girvan = 0
def one_shot_girvan_newman_for_communities(G,graph_layout,node_positions):
    global inside_girvan
    global community_tuples
    
    start_time = time.time()
    # to flag elegxei an exoume ksanaknaei girvan, ayto elegxetai dioti thn prwth fora theloume na parei to grafhma G
    # omws epeidh den svhnoume akmew apo to grafhma mas (opws kanei kai h bivliothikh toy girvan) an ksanakalestei
    # xrhsimopoioume thn plhroforia pou exoume krathsei sta community_tuples
    if (inside_girvan == 0):
        # arxika upologizoume ta connected components tou grafhmatos
        # kai sortaroume analoga me to megethos tous kai ta kanoume reverse
        # wste sthn arxh ths listas na uparxei h sunektikh sunistwsa pou tha kanoume girvan
        sorted_components = sorted(nx.connected_components(G), key=len, reverse=True)
        for i in range(1,len(sorted_components)):   
            community_tuples.append(tuple(sorted_components[i]))
    else:
        sorted_components = sorted(community_tuples, key=len, reverse=True)

    print(sorted_components)
    gcc = sorted_components[0]      # pairnoume thn prwth sunektikh sunistwsa(LC) thn opoia theloume na spasoume(k+1)
    #print("GCC = ",gcc)
    max_conn_component = G.subgraph(gcc).copy()         # kratame thn sunektikh synistwsa me to megalutero megethos
    #print(max_conn_component)

    while(nx.is_connected(max_conn_component)):
        betweenness_dict = cntr.edge_betweenness_centrality(max_conn_component)     # briskoume ta betweeness gia thn sunektikh sunistwsa (d1 =  leksiko)
        #print(betweenness_dict)
        max_betweennsess = max(betweenness_dict, key = betweenness_dict.get)        # akmh me max betweenness
        #print(max_betweennsess)
        max_conn_component.remove_edge(max_betweennsess[0], max_betweennsess[1])    # u, v (oi duo sunektikes sunistwses stis opoies espase)

    sorted_components_subgraph = sorted(nx.connected_components(max_conn_component), key=len, reverse=True)
    # o elegxos autos an einai prwth fora den diagrafei to gcc afou exoume xrhsimopoihsei to G grafhma
    # ara auksanei to metrhth wste tis epomenes fores na xrhsimopoiei ta community_tuples kai na diagrafei thn sunektikh synistwsa pou spasame
    if (inside_girvan == 0):
        inside_girvan += 1
    else :
        community_tuples.remove(gcc)
    # pername ta kainourgia 2 tuples pou dhmiourghthikan sthn arxh
    for i in sorted_components_subgraph:       
        community_tuples.insert(0, tuple(i)) 

    print(community_tuples) 

    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    
######################################################################################################################
#  USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM
######################################################################################################################
count = 1
def use_nx_girvan_newman_for_communities(G,graph_layout,node_positions):
    global count
    global community_tuples

    start_time = time.time()
    flag = 1
    # NA PAIRNEI TA PRWTA COUNT
    community_tuples = []          #na proseksw gt einai topikh metavlhth
    comp = girvan_newman(G)
    print("c = ", next(comp))
    for communities in itertools.islice(comp,count):
        for c in communities:
            #print(comp)
            if(flag >= count):          # gia na mhn sumplhrwnontai ta prohgoumena communitities
                community_tuples.append(tuple(sorted(c)))
            #print(community_tuples)
        flag += 1
    print("###########")
    print("COMMUNITY TUPLES = ",community_tuples)
    print("----------")
    #print(node_positions)
    count += 1
    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    
######################################################################################################################
def divisive_community_detection(G,number_of_divisions,graph_layout,node_positions):

    global hierarchy_of_community_tuples, community_tuples
    
    hierarchy_of_community_tuples = []
    LCi = []
    H = G.copy()                                                                        # gia na mporw na kanw allages panw se grafhma 
    sorted_components = sorted(nx.connected_components(G), key=len, reverse=True)       # arxikopoiw to community tuples symfwna me to parwn grafhma
    for i in range(len(sorted_components)):   
            community_tuples.append(tuple(sorted_components[i]))


    for i in range(number_of_divisions-1):                                              # gt kathe fora diagrafete mia koinothta kai prostithontai 2 
        sorted_components = sorted(nx.connected_components(H), key=len, reverse=True)   # gia na brw thn megalyterh koinothta
        LC = sorted_components[0]                                                       # LC prwto stoixeio ths listas
        print("LC = ", LC)

        LC_Graph = H.subgraph(LC).copy()                                                # (ypo)grafhma megalyterhs koinothtas
        #print("LC_Graph = ", LC_Graph)

        comp = girvan_newman(LC_Graph)                                                  # girvan gia th megalyterh koinothta
        for c in next(comp):
            print("LCi = ", c)
            LCi.append(c)                                                               # pairnoume LC1 kai LC2 pou einai to apotelesma tou Girvan

        
        current_division = [LC] + LCi                                                   # ftiaxnoume th lista [LC, LC1, LC2]
        print("community_tuples = ", community_tuples)
        if tuple(LC) in community_tuples:                                               # ftiaxnw community tuples
            print("mphke ")
            LC_index = community_tuples.index(tuple(LC))                                # thesh megalyterhs koinothtas
            community_tuples.pop(LC_index)                                              # afairw apo community tuples thn megalyetrh koinothta
            community_tuples.append(tuple(LCi[0]))                                      # add sto community tuples ta LC1 kai LC2 pou dhmioyrgountai
            community_tuples.append(tuple(LCi[1]))

        #print("current_division = ", current_division)
        
        hierarchy_of_community_tuples.append(current_division)                          # prosthetoume mia lista pou einai triades [LC, LC1, LC2]

        LCi = []                                                                        # gia thn epomenh epanalhpsh tha exoume diaforetika LC1 kai LC2
        # theloume na digrayoume thn akmh me to megalytero betweeness
        while(nx.is_connected(LC_Graph)):
            betweenness_dict = cntr.edge_betweenness_centrality(LC_Graph)           # briskoume ta betweeness gia thn sunektikh sunistwsa (d1 =  leksiko)
            #print(betweenness_dict)
            max_betweennsess = max(betweenness_dict, key = betweenness_dict.get)    # akmh me max betweenness
            #print(max_betweennsess)
            # den eimai poly sigouros an xreiazontai kai ta 2 remove
            LC_Graph.remove_edge(max_betweennsess[0], max_betweennsess[1])          # u, v (oi duo sunektikes sunistwses stis opoies espase)
            H.remove_edge(max_betweennsess[0], max_betweennsess[1])                 # u, v (oi duo sunektikes sunistwses stis opoies espase)




    end_time = time.time()
    

    print("COMMUNITY TUPLES = ",community_tuples)
    print("hierarchy_of_community_tuples = ", hierarchy_of_community_tuples)
    community_tuples = []                                                           # logo twn append pou kanw pio panw "katharizw" thn lista community tuples

######################################################################################################################
def determine_opt_community_structure(G,hierarchy_of_community_tuples):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "STUDENT_AM_determine_opt_community_structure(G,hierarchy_of_community_tuples)" + bcolors.ENDC +"\n")
   
######################################################################################################################
def add_hamilton_cycle_to_graph(G,node_names_list):

    node_length = len(node_names_list)                          # arxikopoioume to mhkos ths node_names_list
    # h sunarthsh add_edge elegxei an uparxei hdh h akmh gia na mhn thn ksanaprosthesei
    for i in range(node_length-1):
        G.add_edge(node_names_list[i],node_names_list[i+1])     # prosthetoume akmh apo thn korufh pou eimaste sthn epomenh
    G.add_edge(node_names_list[node_length],node_names_list[0]) # theloume hamiltonian cycle ara dn ksexname na epistrefei sto kombo pou ksekinhse
    return G

######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH
######################################################################################################################
def add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY):
    
    # gia na broume tous geitones kathe kombou
    for i in node_names_list:                                               # gia na pairnoume tous geitones kathe kombou
        print(G[i])
        while (True):                                                       # gia na epilegoume tuxaia ena kombo apo tous mh geitonikous kombous
            random_vertice = random.choice(node_names_list)                 # epilegoume tuxaia enan kombo
            if (random_vertice != i and random_vertice not in G[i]):        # na mhn einai o idios kai na mhn anhkei stous geitones
                print(random_vertice)
                break
        for j in range(NUM_RANDOM_EDGES):                                          # prospathoume toses fores na kanoume add thn akmh
            rand_num = random.random()
            print(rand_num)
            if (rand_num < EDGE_ADDITION_PROBABILITY ):                     # an o arithmos pou epilexthike tuxaia einai mikroteros apo thn pithanothta epituxias 
                # h sunarthsh add_edge elegxei an uparxei hdh h akmh gia na mhn thn ksanaprosthesei
                G.add_edge(i,random_vertice)                                # kanoume add thn akmh
                break
    return G
    '''
    1) theloume gia kathe kombo pou exoume sto grafhma na briskoume tous mh geitonikous tou kombous
    2) apo tous mh geitonikous kombous theloume tuxaia na epilegoume enan
    3) se auton pou epileksame theloume na prospathisoume NUM_RANDOM_EDGES fores na kanoume add mia akmh
    4) h kathe fora pou prospathoume na kanoume add mia akmh tha exei pithanothta EDGE_ADDITION_PROBABILITY na ginei add
    '''
######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i

def visualize_communities(G,community_tuples,graph_layout,node_positions):

    #print(community_tuples)
    community_tuples = list(map(list,community_tuples))
    node_positions = []

    node_colors = my_random_color_list_generator(len(community_tuples))  # number_of_communities =  len(community_tuples)
    #print(node_colors)                                                   # epistrefei lista me random xewmata gia kathe community

    color_map = [] 
    for node in G:
        node_index = index_2d(community_tuples, node)                 # se poio comunity brisketai 
        color_node = node_colors[node_index]                          # color gia auto to community
        color_map.append(color_node)                                  # color_map gia to node analoga to comunity pou brisketai
    #print(color_map)

    node_positions = my_graph_plot_routine(G,color_map,'blue','solid',graph_layout,node_positions)

########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G,node_names_list,node_positions = my_menu_graph_construction(G,node_names_list,node_positions)

my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples)