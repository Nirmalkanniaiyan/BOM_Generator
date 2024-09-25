import streamlit as st

import networkx as nx
import matplotlib.pyplot as plt
import random
from datetime import timedelta, datetime
from collections import deque, defaultdict

def random_date():
    start_date = datetime(2020, 1, 1)
    random_days = random.randint(0, 365 * 3)  # Random date within 3 years range
    return start_date + timedelta(days=random_days)

def create_hierarchical_tree(num_product_families, num_product_offerings, num_modules, num_modules_levels, num_parts, num_parts_levels):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Root node (Business Group)
    business_group = "Business Group"
    mfg_date = random_date().strftime('%Y-%m-%d')
    availability = random.randint(50, 1000)
    G.add_node(business_group, manufacturing_date=mfg_date, availability=availability, node_type="business_group")

    # Add Product Families as children of the Business Group
    for i in range(1, num_product_families + 1):
        product_family = f"PF{i}"
        mfg_date = random_date().strftime('%Y-%m-%d')
        availability = random.randint(10, 500)
        G.add_node(product_family, manufacturing_date=mfg_date, availability=availability, node_type="product_family")
        G.add_edge(business_group, product_family, weight=1)

        # Add Product Offerings as children of each Product Family
        for j in range(1, num_product_offerings + 1):
            product_offering = f"PO{i}.{j}"
            mfg_date = random_date().strftime('%Y-%m-%d')
            availability = random.randint(1, 100)
            G.add_node(product_offering, manufacturing_date=mfg_date, availability=availability, node_type="product_offering")
            G.add_edge(product_family, product_offering, weight=1)

            # Add Modules and Sub-Modules
            current_level_nodes = [product_offering]
            
            for level in range(num_modules_levels):
                next_level_nodes = []
                for parent_node in current_level_nodes:
                    # Randomly decide the number of modules at this level
                    num_random_modules = random.randint(1, num_modules)

                    for k in range(1, num_random_modules + 1):
                        module = f"{parent_node}_M{level + 1}.{k}"
                        mfg_date = random_date().strftime('%Y-%m-%d')
                        availability = random.randint(1, 50)
                        G.add_node(module, manufacturing_date=mfg_date, availability=availability, node_type="module")
                        G.add_edge(parent_node, module, weight=random.randint(1, 10))
                        next_level_nodes.append(module)

                        # Add Parts and Sub-Parts
                        part_level_nodes = [module]
                        for part_level in range(num_parts_levels):
                            next_part_level_nodes = []
                            for part_parent in part_level_nodes:
                                num_random_parts = random.randint(1, num_parts)

                                for p in range(1, num_random_parts + 1):
                                    part = f"{part_parent}_P{part_level + 1}.{p}"
                                    mfg_date = random_date().strftime('%Y-%m-%d')
                                    availability = random.randint(1, 30)
                                    G.add_node(part, manufacturing_date=mfg_date, availability=availability, node_type="part")
                                    G.add_edge(part_parent, part, weight=random.randint(1, 100))
                                    next_part_level_nodes.append(part)
                            part_level_nodes = next_part_level_nodes

                current_level_nodes = next_level_nodes

    return G
    
if st.toggle('Show BOM') :
    st.image("Sample_BOM.jpg")

st.title('Creation')

NUM_PRODUCT_FAMILIES = 0
NUM_PRODUCT_OFFERINGS = 0

NUM_MODULES = 0
NUM_MODULES_LEVELS = 0

NUM_PARTS = 0
NUM_PARTS_LEVEL = 0

NUM_PRODUCT_FAMILIES = st.number_input('Number of Product Families', min_value=1, value=1)
NUM_PRODUCT_OFFERINGS = st.number_input('Number of Product Offerings', min_value=1, value=1)

NUM_MODULES = st.number_input('Number of Maximum Modules', min_value=1, value=1)
NUM_MODULES_LEVELS = st.number_input('Number of Module Levels', min_value=1, value=1)

NUM_PARTS = st.number_input('Number of Maximum Parts', min_value=1, value=1)
NUM_PARTS_LEVEL = st.number_input('Number of Parts Levels', min_value=1, value=1)

st.session_state.G = create_hierarchical_tree(NUM_PRODUCT_FAMILIES, NUM_PRODUCT_OFFERINGS, NUM_MODULES, NUM_MODULES_LEVELS, NUM_PARTS, NUM_PARTS_LEVEL)


def visualize_graph(G):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    st.pyplot(plt)

visualize_graph(st.session_state.G)

graph = st.session_state.G
try:
    # Save the graph to GraphML format
    file_name = "graph.graphml"
    nx.write_graphml(graph, file_name)
    
    # Read the saved file as bytes to prepare it for download
    with open(file_name, 'rb') as f:
        file_data = f.read()
    
    # Streamlit component to download the file
    # when the button is clicked prevent reloading the page

    st.download_button(
        label="Download GraphML file",
        data=file_data,
        file_name=file_name,
        mime='application/graphml+xml'
    )

    st.success(f"Graph data saved and ready to download as {file_name}")

except Exception as e:
    st.error(f"An error occurred: {e}")


