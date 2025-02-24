# import streamlit as st
# import geopandas as gpd
# import pandas as pd
# import json
# import plotly.express as px
# import folium
# from folium.plugins import HeatMap
# import networkx as nx
# import numpy as np
# from shapely.geometry import Point, Polygon
# from pyproj import Proj, transform
# from scipy.spatial.distance import pdist, squareform
# from sklearn.cluster import DBSCAN
# from streamlit_folium import folium_static

# # Set up the Streamlit page
# st.set_page_config(layout="wide", page_title="Fire Hydrants & Hazard Zones", page_icon="🚒")
# st.title("🚒 Fire Hydrants & Fire Hazard Zones Analysis")

# # File Paths
# fire_hydrant_geojson = "data/Fire_Hydrants_(DWP).geojson"
# fire_hydrant_csv = "data/Fire_Hydrants_(DWP).csv"
# fire_hazard_geojson = "data/Very_High_Fire_Hazard_Severity_Zones.geojson"

# # Load Fire Hydrant Data
# st.sidebar.subheader("Load Data")
# @st.cache_data
# def load_data():
#     try:
#         hydrant_gdf = gpd.read_file(fire_hydrant_geojson)
#         return hydrant_gdf
#     except:
#         hydrant_df = pd.read_csv(fire_hydrant_csv)
#         hydrant_gdf = gpd.GeoDataFrame(
#             hydrant_df,
#             geometry=gpd.points_from_xy(hydrant_df.Longitude, hydrant_df.Latitude),
#             crs="EPSG:4326"
#         )
#         return hydrant_gdf

# def load_hazard_zones():
#     return gpd.read_file(fire_hazard_geojson)

# hydrant_gdf = load_data()
# hazard_gdf = load_hazard_zones()

# # Convert CRS
# st.sidebar.subheader("Convert CRS to UTM")
# hydrant_gdf = hydrant_gdf.to_crs(epsg=32611)
# hazard_gdf = hazard_gdf.to_crs(epsg=32611)

# # Interactive Plot: Fire Hydrants & Hazard Zones
# st.subheader("🗺 Interactive Fire Hydrants & Hazard Zones Map")
# fig = px.scatter_mapbox(
#     hydrant_gdf.to_crs(epsg=4326),
#     lat=hydrant_gdf.geometry.y,
#     lon=hydrant_gdf.geometry.x,
#     color_discrete_sequence=["red"],
#     size_max=10,
#     zoom=12,
#     title="Fire Hydrants"
# )
# fig.update_layout(mapbox_style="open-street-map")
# st.plotly_chart(fig)

# # Overlay on OpenStreetMap (OSM)
# st.subheader("🌍 Hydrants & Hazard Zones on OpenStreetMap")
# map_osm = folium.Map(location=[hydrant_gdf.geometry.y.mean(), hydrant_gdf.geometry.x.mean()], zoom_start=12)
# folium.GeoJson(hazard_gdf.to_crs(epsg=4326)).add_to(map_osm)
# folium_static(map_osm)

# # Compute Hydrant Distances
# st.subheader("📏 Hydrant Distance Matrix")
# utm_coordinates = np.array([(p.x, p.y) for p in hydrant_gdf.geometry])
# distance_matrix = pd.DataFrame(
#     squareform(pdist(utm_coordinates)),
#     columns=[f"H{i+1}" for i in range(len(utm_coordinates))],
#     index=[f"H{i+1}" for i in range(len(utm_coordinates))]
# )
# st.dataframe(distance_matrix.iloc[:10, :10])

# # Clustering with DBSCAN
# st.subheader("🔹 Hydrant Clustering (DBSCAN)")
# dbscan = DBSCAN(eps=100, min_samples=5)
# hydrant_gdf["cluster"] = dbscan.fit_predict(utm_coordinates)
# fig = px.scatter_mapbox(
#     hydrant_gdf.to_crs(epsg=4326),
#     lat=hydrant_gdf.geometry.y,
#     lon=hydrant_gdf.geometry.x,
#     color=hydrant_gdf["cluster"].astype(str),
#     title="Hydrant Clustering (DBSCAN)",
#     zoom=12,
#     mapbox_style="open-street-map"
# )
# st.plotly_chart(fig)

# # Heatmap of Hydrants
# st.subheader("🔥 Fire Hydrant Density Heatmap")
# hydrant_latlon = [(p.y, p.x) for p in hydrant_gdf.to_crs(epsg=4326).geometry]
# m = folium.Map(location=[hydrant_gdf.geometry.y.mean(), hydrant_gdf.geometry.x.mean()], zoom_start=12)
# HeatMap(hydrant_latlon, radius=15, blur=10, max_zoom=1).add_to(m)
# folium_static(m)

# # Spatial Join to Find Hydrants in Hazard Zones
# st.subheader("🚨 Hydrants Inside Hazard Zones")
# hydrants_in_hazard = gpd.sjoin(hydrant_gdf, hazard_gdf, how="inner", predicate="intersects")
# st.write(f"Total Fire Hydrants in Hazardous Zones: {len(hydrants_in_hazard)}")

# # Plot Fire Hydrants in & Outside Hazardous Zones
# fig = px.scatter_mapbox(
#     hydrants_in_hazard.to_crs(epsg=4326),
#     lat=hydrants_in_hazard.geometry.y,
#     lon=hydrants_in_hazard.geometry.x,
#     color_discrete_sequence=["blue"],
#     zoom=12,
#     title="Fire Hydrants in Hazardous Zones"
# )
# st.plotly_chart(fig)

# # Compute Shortest Paths Between Hydrants
# st.subheader("🔗 Shortest Path Between Hydrants")
# G = nx.Graph()
# for i, (x, y) in enumerate(utm_coordinates):
#     G.add_node(i, pos=(x, y))
# for i in range(len(utm_coordinates)):
#     for j in range(i + 1, len(utm_coordinates)):
#         dist = np.linalg.norm(utm_coordinates[i] - utm_coordinates[j])
#         if dist < 200:
#             G.add_edge(i, j, weight=dist)
# fig = px.scatter(
#     x=[utm_coordinates[i][0] for i in G.nodes],
#     y=[utm_coordinates[i][1] for i in G.nodes],
#     title="Shortest Path Between Fire Hydrants",
#     labels={"x": "X Coordinate (meters)", "y": "Y Coordinate (meters)"}
# )
# st.plotly_chart(fig)

# st.success("✅ Streamlit app with interactive plots ready!")

import streamlit as st
import geopandas as gpd
import pandas as pd
import json
import plotly.express as px
import folium
from folium.plugins import HeatMap
import networkx as nx
import numpy as np
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from streamlit_folium import folium_static

# Set up the Streamlit page
st.set_page_config(layout="wide", page_title="Fire Hydrants & Hazard Zones", page_icon="🚒")
st.title("🚒 Fire Hydrants & Fire Hazard Zones Analysis")

# File Paths
fire_hydrant_geojson = "data/Fire_Hydrants_(DWP).geojson"
fire_hydrant_csv = "data/Fire_Hydrants_(DWP).csv"
fire_hazard_geojson = "data/Very_High_Fire_Hazard_Severity_Zones.geojson"

# Load Fire Hydrant Data
st.sidebar.subheader("Load Data")
@st.cache_data
def load_data():
    try:
        hydrant_gdf = gpd.read_file(fire_hydrant_geojson)
        return hydrant_gdf
    except:
        hydrant_df = pd.read_csv(fire_hydrant_csv)
        hydrant_gdf = gpd.GeoDataFrame(
            hydrant_df,
            geometry=gpd.points_from_xy(hydrant_df.Longitude, hydrant_df.Latitude),
            crs="EPSG:4326"
        )
        return hydrant_gdf

def load_hazard_zones():
    return gpd.read_file(fire_hazard_geojson)

hydrant_gdf = load_data()
hazard_gdf = load_hazard_zones()

# Convert CRS
st.sidebar.subheader("Convert CRS to UTM")
hydrant_gdf = hydrant_gdf.to_crs(epsg=32611)
hazard_gdf = hazard_gdf.to_crs(epsg=32611)

# Interactive Map with Folium
st.subheader("🌍 Interactive Fire Hydrants & Hazard Zones Map")
center = [hydrant_gdf.to_crs(epsg=4326).geometry.y.mean(), hydrant_gdf.to_crs(epsg=4326).geometry.x.mean()]
m = folium.Map(location=center, zoom_start=12)
folium.GeoJson(hazard_gdf.to_crs(epsg=4326), name="Fire Hazard Zones", style_function=lambda x: {"fillColor": "lightblue", "color": "black", "weight": 1}).add_to(m)
for _, row in hydrant_gdf.to_crs(epsg=4326).iterrows():
    folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=3, color='red', fill=True, fill_color='red').add_to(m)
folium_static(m)

# Compute Hydrant Distances
st.subheader("📏 Hydrant Distance Matrix")
utm_coordinates = np.array([(p.x, p.y) for p in hydrant_gdf.geometry])
distance_matrix = pd.DataFrame(
    squareform(pdist(utm_coordinates)),
    columns=[f"H{i+1}" for i in range(len(utm_coordinates))],
    index=[f"H{i+1}" for i in range(len(utm_coordinates))]
)
st.dataframe(distance_matrix.iloc[:10, :10])

# Extract coordinates from geometry
hydrant_gdf["longitude"] = hydrant_gdf.geometry.x
hydrant_gdf["latitude"] = hydrant_gdf.geometry.y

# Ensure valid coordinates
utm_coordinates = np.array(list(zip(hydrant_gdf["longitude"], hydrant_gdf["latitude"])))

# Run DBSCAN clustering
dbscan = DBSCAN(eps=100, min_samples=5)
clusters = dbscan.fit_predict(utm_coordinates)

# Assign cluster labels properly
hydrant_gdf["cluster"] = clusters.astype(str)  # Convert to string for proper visualization

# Check for column existence before plotting
st.subheader("🔹 Hydrant Clustering (DBSCAN)")
if "cluster" in hydrant_gdf.columns:
    fig = px.scatter(
        hydrant_gdf.to_crs(epsg=4326),
        x="longitude",
        y="latitude",
        color="cluster",
        title="Fire Hydrant Clustering",
        labels={"longitude": "Longitude", "latitude": "Latitude"}
    )
    st.plotly_chart(fig)
else:
    st.error("DBSCAN clustering failed. Please check the input data.")


# Heatmap of Hydrants
st.subheader("🔥 Fire Hydrant Density Heatmap")
hydrant_latlon = [(p.y, p.x) for p in hydrant_gdf.to_crs(epsg=4326).geometry]
m_heat = folium.Map(location=center, zoom_start=12)
HeatMap(hydrant_latlon, radius=15, blur=10, max_zoom=1).add_to(m_heat)
folium_static(m_heat)

# Spatial Join to Find Hydrants in Hazard Zones
st.subheader("🚨 Hydrants Inside Hazard Zones")
hydrants_in_hazard = gpd.sjoin(hydrant_gdf, hazard_gdf, how="inner", predicate="intersects")
st.write(f"Total Fire Hydrants in Hazardous Zones: {len(hydrants_in_hazard)}")

# Interactive Plot of Fire Hydrants in & Outside Hazardous Zones
fig_hazard = px.scatter_mapbox(
    hydrants_in_hazard.to_crs(epsg=4326),
    lat=hydrants_in_hazard.geometry.y,
    lon=hydrants_in_hazard.geometry.x,
    color_discrete_sequence=["blue"],
    title="Hydrants in Hazard Zones",
    zoom=12,
    height=500
)
fig_hazard.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_hazard)

# Compute Shortest Paths Between Hydrants
st.subheader("🔗 Shortest Path Between Hydrants")
G = nx.Graph()
for i, (x, y) in enumerate(utm_coordinates):
    G.add_node(i, pos=(x, y))
for i in range(len(utm_coordinates)):
    for j in range(i + 1, len(utm_coordinates)):
        dist = np.linalg.norm(utm_coordinates[i] - utm_coordinates[j])
        if dist < 200:
            G.add_edge(i, j, weight=dist)
shortest_path = nx.shortest_path(G, source=0, target=9, weight="weight")

# Interactive Graph of Shortest Path
fig_path = px.scatter(x=[G.nodes[i]["pos"][0] for i in shortest_path], y=[G.nodes[i]["pos"][1] for i in shortest_path], title="Shortest Path Between Hydrants")
st.plotly_chart(fig_path)

st.success("✅ Interactive Streamlit app ready!")
