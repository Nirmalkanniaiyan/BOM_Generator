import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
from io import BytesIO
from dateutil import parser
from collections import deque, defaultdict
from memory_profiler import memory_usage
import functools
import time


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        mem_before = memory_usage()[0]
        result = func(*args, **kwargs)
        end_time = time.time()
        mem_after = memory_usage()[0]
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before

        st.session_state.setdefault("profiling_results", []).append(
            {
                "function": func.__name__,
                "execution_time": execution_time,
                "memory_used": memory_used,
            }
        )
        return result

    return wrapper


def load_graph(uploaded_file):
    try:
        graph = nx.read_graphml(BytesIO(uploaded_file.read()))
        st.success("Graph loaded successfully!")
        return graph
    except Exception as e:
        st.error(f"An error occurred while loading the graph: {e}")
        return None

def random_date_and_quantity():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    manufacturing_date = start_date + (end_date - start_date) * random.random()
    available_quantity = random.randint(1, 100)
    return manufacturing_date, available_quantity

@profile
def shortest_path_hops(graph, node1, node2):
    try:
        path = nx.shortest_path(graph, source=node1, target=node2)
        num_hops = len(path) - 1
        return path, num_hops
    except nx.NetworkXNoPath:
        return None, float('inf')

@profile
def shortest_path_weights(graph, node1, node2):
    try:
        path = nx.shortest_path(graph, source=node1, target=node2, weight='weight')  
        total_weight = nx.shortest_path_length(graph, source=node1, target=node2, weight='weight')
        return path, total_weight
    except nx.NetworkXNoPath:
        return None, float('inf')

@profile
def are_connected(graph, node1, node2):
    _, num_hops = shortest_path_hops(graph, node1, node2)
    return num_hops > 0

@profile
def enough_raw_materials(graph, product_node, units_needed):
    # st.write(graph.out_edges(product_node, data=True))

    
    def check_parts(node, needed):
        
        available_quantity = graph.nodes[node]['availability']
        st.write(node,"has",available_quantity,"and we need",needed)

        if available_quantity < needed:
            # st.write("not available")
            for parent, child, edge_data in graph.out_edges(node, data=True):
                
                needed_units = edge_data['weight'] * (needed - available_quantity)
                # st.write(child,edge_data['weight'],needed_units)
                if not check_parts(child, needed_units):
                    return False
            
            return False
        
        else :
            # st.write("available")
            return True

    # Start the recursive check from the product node
    return check_parts(product_node, units_needed)

@profile
def check_expiry(graph, product_node):
    try:
        node_data = graph.nodes[product_node]
        if 'manufacturing_date' not in node_data:
            st.error(f"Node {product_node} does not have a 'manufacturing_date' attribute.")
            return False  # Default to not expired
        
        manufacturing_date_str = node_data['manufacturing_date']
        
        try:
            manufacturing_date = parser.parse(manufacturing_date_str)
        except ValueError:
            st.error(f"Manufacturing date '{manufacturing_date_str}' is not in a recognized format.")
            return False  # Default to not expired
        
        current_date = datetime.now()
        expiry_duration = timedelta(days=365*2)  
        time_since_manufacturing = current_date - manufacturing_date
        
        st.write(f"Current Date: {current_date}")
        st.write(f"Manufacturing Date: {manufacturing_date}")
        st.write(f"Time Since Manufacturing: {time_since_manufacturing}")
        
        if time_since_manufacturing > expiry_duration:
            return True  # Expired
        return False  # Not expired
    except Exception as e:
        st.error(f"An error occurred while checking expiry: {e}")
        return False  # Default to not expired in case of error

def visualize_subgraph(G, path):
    subgraph = G.subgraph(path).copy()  # Copy to avoid view issues
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_size=700, node_color="lightgreen", font_size=8, font_weight="bold")
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
    st.pyplot(plt)

def plot_hierarchy(G, root=None):

    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a directed graph (DiGraph).")
    
    if root is None:
        # Automatically select the root node if not provided
        root = [node for node, degree in G.in_degree() if degree == 0]
        if len(root) != 1:
            raise ValueError("Graph must have exactly one root node, or you must provide the root.")
        root = root[0]

    pos = hierarchy_pos(G, root)  # Get hierarchical positions for the graph
    plt.figure(figsize=(10, 8))
    
    # Draw nodes and labels
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
    
    # Draw edges and edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=20)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='black')

    # plt.show()
    st.pyplot(plt)

def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if not isinstance(G, nx.DiGraph):
            raise TypeError('G must be a directed graph.')
        if not children:
            return pos
        dx = width / len(children) 
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

st.title('Graph BOM Synthesizer')

uploaded_file = st.file_uploader("Upload a GraphML file", type=["graphml"])

if uploaded_file is not None:
    # Load the graph from the uploaded file
    G = load_graph(uploaded_file)

    if G is not None:
        # Display basic information about the graph (optional)
        st.write(f"Number of nodes: {G.number_of_nodes()}")
        st.write(f"Number of edges: {G.number_of_edges()}")

        st.header("Graph Visualization")

        # visualize_graph(G)
        # plt = visualize_improved_layout(st.session_state.G)
        plot_hierarchy(G, root="Business Group")

        # st.pyplot(plt)

        options = ["shortest_path_hops", "enough_raw_materials", "check_expiry"]
        option = st.selectbox('Select option', options)

        nodes = list(G.nodes())

        if option == "shortest_path_hops":
            node1 = st.selectbox("Select the first node", nodes, key='shortest_path_hops_node1')
            node2 = st.selectbox("Select the second node", nodes, key='shortest_path_hops_node2')

            if st.button('Find Shortest Path (by Hops)'):
                path, num_hops = shortest_path_hops(G, node1, node2)
                if path:
                    st.write(f"Shortest path between {node1} and {node2}: {path}, Number of hops: {num_hops}")
                    st.write(st.session_state["profiling_results"][-1])
                    
                    st.write("Subgraph of the shortest path:")
                    visualize_subgraph(G, path)  # Visualize the subgraph of the path
                    

                else:
                    st.write(f"No path found between {node1} and {node2}")

            

        elif option == "enough_raw_materials":
            product_node = st.selectbox("Select the product node", nodes, key='enough_raw_materials_product_node')
            units_needed = st.number_input("Enter the units needed", min_value=1, value=1)

            if st.button('Check Raw Materials'):
                enough_materials = enough_raw_materials(G, product_node, units_needed)
                node_properties = G.nodes[product_node]
                st.write(f"Product Node Properties: {node_properties}")
                st.write(st.session_state["profiling_results"][-1])

                if enough_materials:
                    st.write(f"Enough raw materials are available to produce {units_needed} units of product {product_node}.")
                else:
                    st.write(f"Not enough raw materials to produce {units_needed} units of product {product_node}.")

        elif option == "check_expiry":
            product_node_expiry = st.selectbox("Select the product node", nodes, key='check_expiry_product_node')
            if st.button('Check Expiry'):
                expired = check_expiry(G, product_node_expiry)
                st.write(st.session_state["profiling_results"][-1])
                if expired:
                    st.write(f"Product {product_node_expiry} is expired.")
                else:
                    st.write(f"Product {product_node_expiry} is not expired.")