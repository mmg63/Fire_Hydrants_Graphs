# %% [markdown]
# # ✨ Summary
# 
# ✅ Step 1: Load Fire Hydrant & Fire Hazard Zone Data
# 
# ✅ Step 2: Extract Coordinates (from GeoJSON or CSV)
# 
# ✅ Step 3: Convert Latitude/Longitude to UTM (meters)
# 
# ✅ Step 4: Plot hydrants & hazard zones in 2D
# 
# ✅ Step 5: Overlay on OpenStreetMap (OSM)
# 
# ✅ Step 6: Compute distances between hydrants
# 
# ✅ Step 7: Calculate Distances Between Fire Hydrants
# 
# ✅ Step 8: Cluster fire hydrants using DBSCAN
# 
# ✅ Step 9: Compute shortest paths between hydrants using NetworkX
# 
# ✅ Step 10: Generate a heatmap of fire hydrant density
# 

# %% [markdown]
# # Step 1: Load Fire Hydrant Data (CSV or GeoJSON)

# %%
import geopandas as gpd
import pandas as pd
import json
import matplotlib.pyplot as plt
import contextily as ctx  # For OSM Basemap
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
from scipy.spatial.distance import pdist, squareform
import numpy as np

# File Paths
fire_hydrant_geojson = "data/Fire_Hydrants_(DWP).geojson"
fire_hydrant_csv = "data/Fire_Hydrants_(DWP).csv"
fire_hazard_geojson = "data/Very_High_Fire_Hazard_Severity_Zones.geojson"


# %% [markdown]
#  # Step 2: Load Fire Hydrant Locations
# 

# %%
# Try loading hydrants from GeoJSON first
try:
    hydrant_gdf = gpd.read_file(fire_hydrant_geojson)
    print("✅ Fire hydrants loaded from GeoJSON.")
except Exception as e:
    print("⚠️ GeoJSON load failed. Trying CSV...")

    # If GeoJSON fails, load from CSV
    try:
        hydrant_df = pd.read_csv(fire_hydrant_csv)

        # Convert to GeoDataFrame
        hydrant_gdf = gpd.GeoDataFrame(
            hydrant_df,
            geometry=gpd.points_from_xy(hydrant_df.Longitude, hydrant_df.Latitude),
            crs="EPSG:4326"
        )
        print("✅ Fire hydrants loaded from CSV.")

    except Exception as e:
        print("❌ Error loading fire hydrants:", e)
        hydrant_gdf = None


# %% [markdown]
# # Step 3: Load Fire Hazard Zones

# %%
# Load Fire Hazard Zones
hazard_gdf = gpd.read_file(fire_hazard_geojson)
print("✅ Fire Hazard Zones Loaded")


# %% [markdown]
# # Step 4: Convert Latitude/Longitude → UTM (Meters)

# %%
# Define Projections
wgs84 = Proj(init="epsg:4326")  # Latitude/Longitude (WGS84)
utm = Proj(init="epsg:32611")   # UTM Zone 11N (for Los Angeles)

# Convert Fire Hydrant Locations
hydrant_gdf = hydrant_gdf.to_crs(epsg=32611)

# Convert Fire Hazard Zones
hazard_gdf = hazard_gdf.to_crs(epsg=32611)

print("✅ All data converted to UTM projection.")


# %% [markdown]
# # Step 5: Plot Hydrants & Fire Hazard Zones

# %%
# Plot Fire Hydrant Locations & Fire Hazard Zones
fig, ax = plt.subplots(figsize=(12, 10))

# Plot Fire Hazard Zones
hazard_gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.5, label="Fire Hazard Zones")

# Plot Fire Hydrants
hydrant_gdf.plot(ax=ax, color="red", markersize=1, label="Fire Hydrants")

# Formatting
plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.title("Fire Hydrants & Fire Hazard Zones (Projected in UTM)")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# # Step 6: Overlay on OpenStreetMap (OSM)

# %%
fig, ax = plt.subplots(figsize=(12, 10))

# Convert to Web Mercator (EPSG 3857) for OSM
hydrant_gdf.to_crs(epsg=3857).plot(ax=ax, color="red", markersize=1, label="Fire Hydrants")
hazard_gdf.to_crs(epsg=3857).plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.5, label="Fire Hazard Zones")

# Add OpenStreetMap (OSM)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Formatting
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Fire Hydrant Locations & Fire Hazard Zones (OSM)")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# # Step 7: Calculate Distances Between Fire Hydrants

# %%
# Extract UTM coordinates for hydrants
utm_coordinates = np.array([(p.x, p.y) for p in hydrant_gdf.geometry])

# Compute pairwise Euclidean distances (meters)
distance_matrix = pd.DataFrame(
    squareform(pdist(utm_coordinates)), 
    columns=[f"H{i+1}" for i in range(len(utm_coordinates))], 
    index=[f"H{i+1}" for i in range(len(utm_coordinates))]
)

print("✅ Distance matrix calculated (meters). Showing first 5 rows:")
print(distance_matrix.iloc[:5, :5])


# %% [markdown]
# # 🔹 Step 8: Cluster Fire Hydrants Using DBSCAN
# 
# We'll use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to group nearby hydrants into clusters.

# %%
from sklearn.cluster import DBSCAN
import seaborn as sns

# DBSCAN Clustering
dbscan = DBSCAN(eps=100, min_samples=5)  # eps in meters (100m radius)
hydrant_gdf["cluster"] = dbscan.fit_predict(utm_coordinates)

# Plot Clusters
fig, ax = plt.subplots(figsize=(12, 10))
hydrant_gdf.plot(ax=ax, column="cluster", cmap="viridis", markersize=1, legend=True)
hazard_gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.5, label="Fire Hazard Zones")

plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.title("Fire Hydrant Clusters (DBSCAN)")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# # 🔹 Step 9: Compute Shortest Paths Between Fire Hydrants
# 
# We'll build a graph network and find the shortest path between hydrants.

# %%
import networkx as nx

# Create Graph from Hydrant Locations
G = nx.Graph()

# Add Nodes (Hydrant Positions)
for i, (x, y) in enumerate(utm_coordinates):
    G.add_node(i, pos=(x, y))

# Add Edges (Connecting Nearby Hydrants)
for i in range(len(utm_coordinates)):
    for j in range(i + 1, len(utm_coordinates)):
        dist = np.linalg.norm(utm_coordinates[i] - utm_coordinates[j])
        if dist < 200:  # Connect hydrants within 200 meters
            G.add_edge(i, j, weight=dist)

# Compute Shortest Path Between Two Hydrants (Example: H1 to H10)
shortest_path = nx.shortest_path(G, source=0, target=9, weight="weight")

# Plot Shortest Path
fig, ax = plt.subplots(figsize=(12, 10))
nx.draw(G, pos=nx.get_node_attributes(G, "pos"), node_color="red", edge_color="gray", node_size=30, ax=ax)
nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G, "pos"), nodelist=shortest_path, node_color="blue", node_size=50, ax=ax)
nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, "pos"), edgelist=[(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)], edge_color="blue", width=2, ax=ax)

plt.title("Shortest Path Between Fire Hydrants")
plt.show()


# %% [markdown]
# # 🔹 Step 10: Generate a Heatmap of Fire Hydrant Density

# %%
import folium
from folium.plugins import HeatMap

# Convert hydrant locations to list of lat/lon
hydrant_latlon = [(p.y, p.x) for p in hydrant_gdf.to_crs(epsg=4326).geometry]

# Create Base Map
m = folium.Map(location=[np.mean([p.y for p in hydrant_gdf.geometry]), np.mean([p.x for p in hydrant_gdf.geometry])], zoom_start=12)

# Add Heatmap Layer
HeatMap(hydrant_latlon, radius=15, blur=10, max_zoom=1).add_to(m)

# Save and Display Map
m.save("fire_hydrant_heatmap.html")
m


# %% [markdown]
# # Step 11.1: Perform a Spatial Join Between Hydrants & Fire Hazard Zones

# %%
# Ensure both datasets have the same CRS (Coordinate Reference System)
hydrant_gdf = hydrant_gdf.to_crs(hazard_gdf.crs)

# Perform Spatial Join: Find hydrants inside hazard zones
hydrants_in_hazard = gpd.sjoin(hydrant_gdf, hazard_gdf, how="inner", predicate="within")

# Display results
print(f"Total Fire Hydrants in Hazardous Zones: {len(hydrants_in_hazard)}")
hydrants_in_hazard.head()


# %% [markdown]
# # Step 11.2: Plot Fire Hydrants in & Outside Hazardous Zones

# %%
fig, ax = plt.subplots(figsize=(12, 10))

# Plot Fire Hazard Zones
hazard_gdf.plot(ax=ax, edgecolor="black", facecolor="lightcoral", alpha=0.5, label="Fire Hazard Zones")

# Plot Fire Hydrants Inside Hazard Zones
hydrants_in_hazard.plot(ax=ax, color="blue", markersize=1, label="Hydrants in Hazard Zones")

# Plot Fire Hydrants Outside Hazard Zones
hydrant_gdf[~hydrant_gdf.index.isin(hydrants_in_hazard.index)].plot(ax=ax, color="green", markersize=1, label="Safe Hydrants")

# Formatting
plt.xlabel("X Coordinate (meters)")
plt.ylabel("Y Coordinate (meters)")
plt.title("Fire Hydrants Inside and Outside Hazard Zones")
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# # Step 11.3: Percentage of Fire Hydrants in Hazardous Zones

# %%
# Calculate percentage of hydrants in hazard zones
total_hydrants = len(hydrant_gdf)
hazard_hydrants = len(hydrants_in_hazard)

percentage_in_hazard = (hazard_hydrants / total_hydrants) * 100
print(f"Percentage of Fire Hydrants in Hazard Zones: {percentage_in_hazard:.2f}%")


# %%



