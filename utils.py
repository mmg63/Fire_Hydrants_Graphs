import numpy as np
import geopandas as gpd

def get_hydrant_metric_space(hydrant_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Convert fire hydrant locations from a GeoDataFrame to a metric space (UTM coordinates).
    
    Parameters:
    hydrant_gdf (gpd.GeoDataFrame): A GeoDataFrame containing fire hydrant locations with geometry column.
    
    Returns:
    np.ndarray: A NumPy array of (x, y) coordinates in UTM.
    """
    if hydrant_gdf.crs is None:
        raise ValueError("The input GeoDataFrame must have a defined CRS.")
    
    if hydrant_gdf.crs.to_epsg() != 32611:
        hydrant_gdf = hydrant_gdf.to_crs(epsg=32611)
    
    return np.array([(p.x, p.y) for p in hydrant_gdf.geometry])
