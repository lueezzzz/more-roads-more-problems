# More Roads, More Problems

Welcome to the official repository for our thesis, **"More Roads, More Problems."** This repository has been compiled and made publicly available to serve as a resource for future researchers, students, and developers. Whether you are looking to replicate our findings, explore our methodology, or build upon our work for your own studies, you will find the necessary code, scripts, and documentation here.

## Purpose

The primary goal of this repository is to ensure transparency and reproducibility in our research. We hope that providing open access to our workflow will assist others in tackling similar challenges and advancing the research in this field.

## Project Structure

### `iloilo-districts-python-files/`

Contains Python files for different districts of Iloilo City. Each file is named after a specific district and contains district-specific network data:

- `arevalo.py` - Arevalo district
- `city_proper.py` - City Proper district
- `jaro.py` - Jaro district
- `lapaz.py` - La Paz district
- `lapuz.py` - Lapuz district
- `mandurriao.py` - Mandurriao district
- `molo.py` - Molo district

### `network-visualizer/`

Contains scripts and data for visualizing the road network:

- `visualizer.py` - Main visualization script
- `final_network.py` - Final network configuration and generation
- `roads.csv` - Road segments data
- `intersections.csv` - Intersection nodes data

### `parameters/`

Contains parameter files used for network analysis and simulation:

- `roads_updated.csv` - Updated road data with parameters
- `intersections.csv` - Intersection data with parameters
- `zones.csv` - Zone data for the analysis

### `setups/`

Contains Python scripts implementing different network analysis setups and configurations:

- `uni aadt.py` - Unidirectional analysis based on AADT (Annual Average Daily Traffic)
- `uni centroid.py` - Unidirectional analysis based on centroid distribution
- `uni distributed.py` - Unidirectional analysis with distributed traffic
- `bi aadt.py` - Bidirectional analysis based on AADT
- `bi centroid.py` - Bidirectional analysis based on centroid distribution
- `bi distributed.py` - Bidirectional analysis with distributed traffic
- `all pair.py` - All-pairs analysis

### `setup-results/`

Contains CSV files with results from each setup's analysis:

- `All Pairs Results.csv` - Results from all-pairs analysis
- `Setup 1-6 Results.csv` - Individual results for each setup
- `Setup 1-6 Baseline Network Flows.csv` - Baseline flow data for each setup
