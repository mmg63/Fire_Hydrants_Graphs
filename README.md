# Fire Hydrants & Fire Hazard Zones Analysis

🚒 **Fire Hydrants & Fire Hazard Zones Analysis** is a Streamlit web application designed to visualize and analyze fire hydrant locations and fire hazard severity zones. This project integrates GIS data, spatial clustering, shortest path computation, and interactive mapping to support fire safety analysis.

## 📌 Features
- **Interactive Mapping**: Uses `folium` to visualize fire hydrants and hazard zones on OpenStreetMap.
- **Clustering with DBSCAN**: Groups nearby hydrants for analysis using DBSCAN clustering.
- **Distance Matrix Computation**: Calculates distances between hydrants.
- **Shortest Path Computation**: Uses `networkx` to determine the shortest path between hydrants.
- **Fire Hydrant Density Heatmap**: Visualizes hydrant density using `folium.plugins.HeatMap`.
- **Spatial Joins**: Identifies fire hydrants located inside fire hazard zones.
- **Interactive Plots**: Uses `plotly` for dynamic data visualization.

## 📂 File Structure
```
project_directory/
│-- data/
│   ├── Fire_Hydrants_(DWP).geojson
│   ├── Fire_Hydrants_(DWP).csv
│   ├── Very_High_Fire_Hazard_Severity_Zones.geojson
│-- 20250223_Main_Streamlit_V1.py
│-- README.md
```

## 🛠 Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ and the required libraries installed:
```bash
pip install streamlit geopandas pandas folium networkx numpy shapely pyproj scipy scikit-learn plotly contextily streamlit_folium
```

### Running the App
To launch the Streamlit application, use:
```bash
streamlit run 20250223_Main_Streamlit_V1.py
```
This will open the web app in a new browser tab.

## 📊 Data Sources
- **Fire Hydrant Data**: Comes from a GeoJSON/CSV file with latitude and longitude information.
- **Fire Hazard Zones**: GIS data indicating high-risk fire zones.

## 🔍 Usage Guide
1. **Load Data**: The app automatically loads hydrant and hazard zone data.
2. **View Interactive Maps**: Fire hydrants and hazard zones are displayed on an interactive folium map.
3. **Analyze Hydrant Clusters**: The DBSCAN algorithm identifies clusters of hydrants.
4. **Heatmap Analysis**: Identifies fire hydrant density distribution.
5. **Find Shortest Paths**: Computes shortest paths between hydrants for emergency response planning.
6. **View Distance Matrix**: Hydrant distances are displayed in a tabular format.

## 🏗 Future Improvements
- Integration with real-time GIS data.
- Enhanced predictive analytics for fire risk assessment.
- User-friendly UI enhancements.

## 📝 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Contributions are welcome! If you have ideas or fixes, feel free to submit a pull request.

## 📧 Contact
For any questions, please contact the project maintainers or raise an issue on the repository.

