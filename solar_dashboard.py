import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import datetime
from datetime import datetime, timedelta
import warnings
import folium
from streamlit_folium import st_folium
import requests
import json
import time
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Solar Energy Production Prediction System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        color: #28A745;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #E8F5E8;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #000;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .geo-info-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFA726;
        margin: 1rem 0;
    }
    .map-container {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Weather API integration functions
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_weather_data(lat, lon, api_key=None):
    """
    Get current weather data from OpenWeatherMap API
    Falls back to mock data if API key is not provided or API fails
    """
    if api_key:
        try:
            # Current weather API call
            current_url = f"https://api.openweathermap.org/data/2.5/weather"
            current_params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            current_response = requests.get(current_url, params=current_params, timeout=10)
            
            if current_response.status_code == 200:
                current_data = current_response.json()
                
                # Extract relevant weather data
                weather_data = {
                    'temperature': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'wind_speed': current_data['wind']['speed'],
                    'pressure': current_data['main']['pressure'],
                    'clouds': current_data['clouds']['all'],
                    'visibility': current_data.get('visibility', 10000) / 1000,  # Convert to km
                    'weather_main': current_data['weather'][0]['main'],
                    'weather_description': current_data['weather'][0]['description'],
                    'sunrise': datetime.fromtimestamp(current_data['sys']['sunrise']),
                    'sunset': datetime.fromtimestamp(current_data['sys']['sunset']),
                    'location': f"{current_data['name']}, {current_data['sys']['country']}",
                    'last_updated': datetime.now(),
                    'data_source': 'OpenWeatherMap API'
                }
                
                # Calculate solar irradiation estimate based on weather conditions
                base_irradiation = 800  # Base solar irradiation on clear day
                cloud_factor = (100 - weather_data['clouds']) / 100
                weather_factor = 1.0
                
                if weather_data['weather_main'] in ['Rain', 'Drizzle', 'Thunderstorm']:
                    weather_factor = 0.3
                elif weather_data['weather_main'] in ['Snow', 'Mist', 'Fog']:
                    weather_factor = 0.5
                elif weather_data['weather_main'] == 'Clouds':
                    weather_factor = 0.7
                
                weather_data['estimated_irradiation'] = base_irradiation * cloud_factor * weather_factor
                
                return weather_data
                
        except Exception as e:
            st.warning(f"Weather API error: {str(e)}. Using simulated data.")
    
    # Fallback to mock data
    return generate_mock_weather_data(lat, lon)

def generate_mock_weather_data(lat, lon):
    """Generate realistic mock weather data based on location"""
    # Simulate weather based on location (very basic)
    base_temp = 25 + (lat * 0.5)  # Rough temperature based on latitude
    
   # Add some randomness but keep it realistic
    np.random.seed(int(time.time()) % 1000)
    
    current_hour = datetime.now().hour
    
    # Daily temperature variation
    temp_variation = 5 * np.sin((current_hour - 6) * np.pi / 12)
    
    weather_data = {
        'temperature': base_temp + temp_variation + np.random.normal(0, 2),
        'humidity': np.random.uniform(40, 80),
        'wind_speed': np.random.uniform(1, 12),
        'pressure': np.random.uniform(1010, 1025),
        'clouds': np.random.uniform(0, 100),
        'visibility': np.random.uniform(5, 15),
        'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.6, 0.3, 0.1]),
        'weather_description': 'simulated weather conditions',
        'sunrise': datetime.now().replace(hour=6, minute=30),
        'sunset': datetime.now().replace(hour=18, minute=30),
        'location': f"Location ({lat:.2f}, {lon:.2f})",
        'last_updated': datetime.now(),
        'data_source': 'Simulated Data'
    }
    
    # Calculate estimated irradiation
    base_irradiation = 800
    cloud_factor = (100 - weather_data['clouds']) / 100
    weather_factor = 0.3 if weather_data['weather_main'] == 'Rain' else 0.7 if weather_data['weather_main'] == 'Clouds' else 1.0
    
    weather_data['estimated_irradiation'] = base_irradiation * cloud_factor * weather_factor
    
    return weather_data

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_forecast(lat, lon, api_key=None):
    """
    Get 5-day weather forecast from OpenWeatherMap API
    Falls back to mock data if API key is not provided
    """
    if api_key:
        try:
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast"
            forecast_params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                
                forecast_list = []
                for item in forecast_data['list']:
                    forecast_point = {
                        'datetime': datetime.fromtimestamp(item['dt']),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'wind_speed': item['wind']['speed'],
                        'clouds': item['clouds']['all'],
                        'weather_main': item['weather'][0]['main'],
                        'weather_description': item['weather'][0]['description']
                    }
                    
                    # Estimate irradiation
                    base_irradiation = 800
                    cloud_factor = (100 - forecast_point['clouds']) / 100
                    weather_factor = 0.3 if forecast_point['weather_main'] == 'Rain' else 0.7 if forecast_point['weather_main'] == 'Clouds' else 1.0
                    forecast_point['estimated_irradiation'] = base_irradiation * cloud_factor * weather_factor
                    
                    forecast_list.append(forecast_point)
                
                return forecast_list
                
        except Exception as e:
            st.warning(f"Forecast API error: {str(e)}. Using simulated data.")
    
    # Fallback to mock forecast data
    return generate_mock_forecast_data(lat, lon)

def generate_mock_forecast_data(lat, lon):
    """Generate mock forecast data"""
    forecast_list = []
    base_temp = 25 + (lat * 0.5)
    
    for i in range(40):  # 5 days, 8 forecasts per day (3-hour intervals)
        forecast_time = datetime.now() + timedelta(hours=i*3)
        
        # Daily temperature variation
        hour = forecast_time.hour
        temp_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        
        np.random.seed(int(forecast_time.timestamp()) % 1000)
        
        forecast_point = {
            'datetime': forecast_time,
            'temperature': base_temp + temp_variation + np.random.normal(0, 2),
            'humidity': np.random.uniform(40, 80),
            'wind_speed': np.random.uniform(1, 12),
            'clouds': np.random.uniform(0, 100),
            'weather_main': np.random.choice(['Clear', 'Clouds', 'Rain'], p=[0.6, 0.3, 0.1]),
            'weather_description': 'simulated forecast'
        }
        
        # Estimate irradiation
        base_irradiation = 800
        cloud_factor = (100 - forecast_point['clouds']) / 100
        weather_factor = 0.3 if forecast_point['weather_main'] == 'Rain' else 0.7 if forecast_point['weather_main'] == 'Clouds' else 1.0
        forecast_point['estimated_irradiation'] = base_irradiation * cloud_factor * weather_factor
        
        forecast_list.append(forecast_point)
    
    return forecast_list

def calculate_module_temperature(ambient_temp, irradiation, wind_speed):
    """Calculate module temperature based on ambient conditions"""
    # Simplified NOCT (Nominal Operating Cell Temperature) calculation
    # Module temp = Ambient temp + (NOCT - 20) * (Irradiation / 800) / (1 + wind_speed/5)
    noct = 45  # Typical NOCT for silicon panels
    module_temp = ambient_temp + (noct - 20) * (irradiation / 800) / (1 + wind_speed / 5)
    return module_temp

# Generate synthetic data for demonstration
@st.cache_data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic solar power data for demonstration"""
    np.random.seed(42)
    
    # Generate time series data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate synthetic weather features
    data = {
        'DATE_TIME': dates,
        'AMBIENT_TEMPERATURE': np.random.normal(25, 8, n_samples),
        'MODULE_TEMPERATURE': np.random.normal(35, 10, n_samples),
        'IRRADIATION': np.random.uniform(0, 1000, n_samples),
        'HUMIDITY': np.random.uniform(20, 80, n_samples),
        'WIND_SPEED': np.random.uniform(0, 15, n_samples)
    }
    
    # Generate synthetic power output based on realistic relationships
    df = pd.DataFrame(data)
    df['HOUR'] = pd.to_datetime(df['DATE_TIME']).dt.hour
    df['MONTH'] = pd.to_datetime(df['DATE_TIME']).dt.month
    df['DAY_OF_YEAR'] = pd.to_datetime(df['DATE_TIME']).dt.dayofyear
    
    # Realistic power generation based on irradiation and temperature
    df['DC_POWER'] = (
        df['IRRADIATION'] * 0.05 +  # Primary factor
        df['AMBIENT_TEMPERATURE'] * 0.1 +  # Secondary factor
        np.sin(df['HOUR'] * np.pi / 12) * 20 +  # Daily pattern
        np.random.normal(0, 5, n_samples)  # Noise
    )
    
    # Ensure no negative power values
    df['DC_POWER'] = np.maximum(df['DC_POWER'], 0)
    
    # AC power is typically slightly less than DC power
    df['AC_POWER'] = df['DC_POWER'] * 0.95
    
    return df

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    """Load and prepare the solar power data"""
    df = generate_synthetic_data(2000)
    
    # Feature engineering
    df['HOUR'] = pd.to_datetime(df['DATE_TIME']).dt.hour
    df['MONTH'] = pd.to_datetime(df['DATE_TIME']).dt.month
    df['DAY_OF_YEAR'] = pd.to_datetime(df['DATE_TIME']).dt.dayofyear
    df['TEMP_IRRADIATION'] = df['AMBIENT_TEMPERATURE'] * df['IRRADIATION']
    df['IS_DAYTIME'] = ((df['HOUR'] >= 6) & (df['HOUR'] <= 18)).astype(int)
    
    return df

# Train model
@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 
                'HUMIDITY', 'WIND_SPEED', 'HOUR', 'MONTH', 'DAY_OF_YEAR', 
                'TEMP_IRRADIATION', 'IS_DAYTIME']
    
    X = df[features]
    y = df['DC_POWER']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, rmse, r2, X_test, y_test, y_pred, features

# Geological data and site suitability assessment
@st.cache_data
def get_geological_info(lat, lon):
    """Get geological information for a given location"""
    # Mock geological data - in reality, you'd use APIs like USGS or geological survey services
    geological_types = [
        "Sedimentary Rock", "Igneous Rock", "Metamorphic Rock", 
        "Alluvial Soil", "Clay Soil", "Sandy Soil", "Rocky Terrain"
    ]
    
    # Simulate geological data based on location
    np.random.seed(int(lat * 1000 + lon * 1000))
    geology_type = np.random.choice(geological_types)
    
    # Simulate soil properties
    soil_stability = np.random.uniform(0.6, 0.95)
    drainage_quality = np.random.uniform(0.5, 0.9)
    foundation_suitability = np.random.uniform(0.7, 0.95)
    
    # Calculate overall suitability score
    suitability_score = (soil_stability * 0.4 + drainage_quality * 0.3 + foundation_suitability * 0.3)
    
    return {
        'geology_type': geology_type,
        'soil_stability': soil_stability,
        'drainage_quality': drainage_quality,
        'foundation_suitability': foundation_suitability,
        'suitability_score': suitability_score
    }

# Create geological map
def create_geological_map(center_lat=0, center_lon=0):
    """Create a geological map with different layers"""
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers for geological context
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add geological markers (simulated data)
    geological_sites = [
        {'lat': center_lat + 0.1, 'lon': center_lon + 0.1, 'type': 'Sedimentary', 'suitability': 0.85},
        {'lat': center_lat - 0.1, 'lon': center_lon + 0.1, 'type': 'Igneous', 'suitability': 0.75},
        {'lat': center_lat + 0.1, 'lon': center_lon - 0.1, 'type': 'Alluvial', 'suitability': 0.90},
        {'lat': center_lat - 0.1, 'lon': center_lon - 0.1, 'type': 'Clay Soil', 'suitability': 0.65},
    ]
    
    for site in geological_sites:
        # Color based on suitability
        if site['suitability'] > 0.8:
            color = 'green'
        elif site['suitability'] > 0.7:
            color = 'orange'
        else:
            color = 'red'
        
        folium.CircleMarker(
            location=[site['lat'], site['lon']],
            radius=8,
            popup=f"Type: {site['type']}<br>Suitability: {site['suitability']:.2f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Site Suitability</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Excellent (>0.8)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Good (0.7-0.8)</p>
    <p><i class="fa fa-circle" style="color:red"></i> Fair (<0.7)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ Solar Energy Production Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About this System")
        st.markdown("""
        This application predicts solar energy production based on weather conditions and geological factors.
        
        **Features:**
        - Real-time prediction based on weather inputs
        - Interactive geological maps
        - Site suitability assessment
        - Model performance metrics
        - SDG 7 alignment: Affordable and Clean Energy
        """)
        
        st.markdown("### How to Use")
        st.markdown("""
        1. Select location on the geological map
        2. Adjust weather parameters
        3. Click 'Predict' to get power output prediction
        4. View geological site assessment
        5. Explore visualizations and insights
        """)

         # Weather API settings
        st.markdown("### 🌤️ Weather API Settings")
        use_real_weather = st.checkbox("Use Real Weather Data", value=False)
        
        if use_real_weather:
            api_key = st.text_input(
                "OpenWeatherMap API Key", 
                type="password",
                help="Get your free API key from openweathermap.org"
            )
            if not api_key:
                st.warning("⚠️ Please enter your OpenWeatherMap API key to use real weather data.")
        else:
            api_key = None
            st.info("💡 Using simulated weather data. Enable real weather data above.")
        
        # Weather data refresh
        if st.button("🔄 Refresh Weather Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Location input
        st.markdown("### 📍 Location Settings")
        latitude = st.number_input("Latitude", value=-1.2921, format="%.6f")
        longitude = st.number_input("Longitude", value=36.8219, format="%.6f")
        
        # Get geological info for selected location
        geo_info = get_geological_info(latitude, longitude)

        # Get real-time weather data
        weather_data = get_weather_data(latitude, longitude, api_key)

         # Display current weather conditions
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🌤️ Current Weather**")
        st.markdown(f"**Location:** {weather_data['location']}")
        st.markdown(f"**Temperature:** {weather_data['temperature']:.1f}°C")
        st.markdown(f"**Humidity:** {weather_data['humidity']:.0f}%")
        st.markdown(f"**Wind Speed:** {weather_data['wind_speed']:.1f} m/s")
        st.markdown(f"**Clouds:** {weather_data['clouds']:.0f}%")
        st.markdown(f"**Conditions:** {weather_data['weather_description'].title()}")
        st.markdown(f"**Est. Irradiation:** {weather_data['estimated_irradiation']:.0f} W/m²")
        st.markdown(f"**Updated:** {weather_data['last_updated'].strftime('%H:%M')}")
        st.markdown(f"**Source:** {weather_data['data_source']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="geo-info-box">', unsafe_allow_html=True)
        st.markdown("**🌍 Geological Assessment**")
        st.markdown(f"**Geology Type:** {geo_info['geology_type']}")
        st.markdown(f"**Soil Stability:** {geo_info['soil_stability']:.2f}")
        st.markdown(f"**Drainage Quality:** {geo_info['drainage_quality']:.2f}")
        st.markdown(f"**Foundation Suitability:** {geo_info['foundation_suitability']:.2f}")
        
        # Overall suitability
        suitability = geo_info['suitability_score']
        if suitability > 0.8:
            st.success(f"**Overall Suitability:** {suitability:.2f} (Excellent)")
        elif suitability > 0.7:
            st.info(f"**Overall Suitability:** {suitability:.2f} (Good)")
        else:
            st.warning(f"**Overall Suitability:** {suitability:.2f} (Fair)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data and train model
    df = load_and_prepare_data()
    model, rmse, r2, X_test, y_test, y_pred, features = train_model(df)
    
     # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction System", "🌤️ Weather Dashboard", "🗺️ Geological Maps", "📊 Analysis & Insights"])
    
    with tab1:
        # Layout: Three columns
        col1, col2, col3 = st.columns([1, 1, 2])
        
        # Column 1: Input Parameters
        with col1:
            st.markdown('<h3 class="sub-header">🌡️ Input Weather Parameters</h3>', 
                        unsafe_allow_html=True)
            
           # Option to use real weather data or manual input
            use_current_weather = st.checkbox("Use Current Weather Data", value=True)
            
            if use_current_weather:
                # Use real weather data
                temperature = weather_data['temperature']
                humidity = weather_data['humidity']
                wind_speed = weather_data['wind_speed']
                irradiation = weather_data['estimated_irradiation']
                
                # Calculate module temperature
                module_temp = calculate_module_temperature(temperature, irradiation, wind_speed)
                
                # Display current values (read-only)
                st.metric("Current Temperature", f"{temperature:.1f}°C")
                st.metric("Current Humidity", f"{humidity:.0f}%")
                st.metric("Current Wind Speed", f"{wind_speed:.1f} m/s")
                st.metric("Estimated Irradiation", f"{irradiation:.0f} W/m²")
                st.metric("Calculated Module Temp", f"{module_temp:.1f}°C")
                
                st.info("📡 Using real-time weather data. Uncheck to use manual inputs.")
                
            else:
                # Manual weather input sliders
                st.markdown("**Manual Weather Input**")
                
                temperature = st.slider(
                    "Ambient Temperature (°C)", 
                    min_value=0.0, 
                    max_value=50.0, 
                    value=float(weather_data['temperature']),
                    help="Average ambient temperature"
                )
                
                module_temp = st.slider(
                    "Module Temperature (°C)", 
                    min_value=0.0, 
                    max_value=60.0, 
                    value=35.0,
                    help="Solar panel surface temperature"
                )
                
                irradiation = st.slider(
                    "Solar Irradiation (W/m²)", 
                    min_value=0.0, 
                    max_value=1000.0, 
                    value=float(weather_data['estimated_irradiation']),
                    help="Solar irradiation intensity"
                )
                
                humidity = st.slider(
                    "Humidity (%)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=float(weather_data['humidity']),
                    help="Relative humidity"
                )
                
                wind_speed = st.slider(
                    "Wind Speed (m/s)", 
                    min_value=0.0, 
                    max_value=20.0, 
                    value=float(weather_data['wind_speed']),
                    help="Wind speed"
                )
            
            # Time inputs
            st.markdown("**⏰ Time Parameters**")
            current_time = datetime.now()
            hour = st.slider("Hour of Day", 0, 23, current_time.hour)
            month = st.selectbox("Month", range(1, 13), index=current_time.month-1)
            day_of_year = st.slider("Day of Year", 1, 365, current_time.timetuple().tm_yday)

        
        # Column 2: Prediction Results
        with col2:
            st.markdown('<h3 class="sub-header">🔮 Prediction Results</h3>', 
                        unsafe_allow_html=True)
            
            # Prepare input data
            temp_irradiation = temperature * irradiation
            is_daytime = 1 if 6 <= hour <= 18 else 0
            
            input_data = pd.DataFrame({
                'AMBIENT_TEMPERATURE': [temperature],
                'MODULE_TEMPERATURE': [module_temp],
                'IRRADIATION': [irradiation],
                'HUMIDITY': [humidity],
                'WIND_SPEED': [wind_speed],
                'HOUR': [hour],
                'MONTH': [month],
                'DAY_OF_YEAR': [day_of_year],
                'TEMP_IRRADIATION': [temp_irradiation],
                'IS_DAYTIME': [is_daytime]
            })
            
            # Make prediction
            if st.button("🚀 Predict Power Output", type="primary"):
                prediction = model.predict(input_data)[0]
                
                # Adjust prediction based on geological suitability
                geo_adjusted_prediction = prediction * geo_info['suitability_score']
                
                st.markdown(f'''
                <div class="prediction-result">
                    ⚡ {prediction:.2f} kW
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown(f'''
                <div class="info-box">
                    <strong>Geo-Adjusted Prediction:</strong><br>
                    {geo_adjusted_prediction:.2f} kW<br>
                    <small>(Adjusted for geological conditions)</small>
                </div>
                ''', unsafe_allow_html=True)
                
                # Additional insights
                if prediction > 30:
                    st.success("High power output expected! ☀️")
                elif prediction > 15:
                    st.info("Moderate power output expected 🌤️")
                else:
                    st.warning("Low power output expected ⛅")
            
            # Model Performance Metrics
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown("**📊 Model Performance**")
            st.metric("RMSE", f"{rmse:.2f} kW")
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("Accuracy", f"{r2*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # SDG Impact
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**🌍 SDG 7 Impact**")
            st.markdown("This system supports **Affordable and Clean Energy** by optimizing solar power planning and reducing fossil fuel dependency.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Column 3: Quick Visualizations
        with col3:
            st.markdown('<h3 class="sub-header">📈 Quick Insights</h3>', 
                        unsafe_allow_html=True)
            
            # Historical power output time series
            fig_timeseries = px.line(
                df.head(100),
                x='DATE_TIME',
                y='DC_POWER',
                title="Recent Solar Power Output Trend"
            )
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # Power output by hour of day
            hourly_avg = df.groupby('HOUR')['DC_POWER'].mean().reset_index()
            fig_hourly = px.bar(
                hourly_avg,
                x='HOUR',
                y='DC_POWER',
                title="Average Power Output by Hour"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

    with tab2:
        st.markdown('<h3 class="sub-header">🌤️ Weather Dashboard</h3>', 
                    unsafe_allow_html=True)
        
        # Get forecast data
        forecast_data = get_weather_forecast(latitude, longitude, api_key)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Weather forecast chart
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                
                # Temperature and irradiation forecast
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['datetime'],
                    y=forecast_df['temperature'],
                    mode='lines+markers',
                    name='Temperature (°C)',
                    line=dict(color='red'),
                    yaxis='y'
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['datetime'],
                    y=forecast_df['estimated_irradiation'],
                    mode='lines+markers',
                    name='Est. Irradiation (W/m²)',
                    line=dict(color='orange'),
                    yaxis='y2'
                ))
                
                fig_forecast.update_layout(
                    title='5-Day Weather Forecast',
                    xaxis_title='Date & Time',
                    yaxis=dict(title='Temperature (°C)', side='left'),
                    yaxis2=dict(title='Irradiation (W/m²)', side='right', overlaying='y'),
                    legend=dict(x=0, y=1),
                    height=400
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Humidity and wind speed forecast
                fig_conditions = go.Figure()
                
                fig_conditions.add_trace(go.Scatter(
                    x=forecast_df['datetime'],
                    y=forecast_df['humidity'],
                    mode='lines+markers',
                    name='Humidity (%)',
                    line=dict(color='blue'),
                    yaxis='y'
                ))
                
                fig_conditions.add_trace(go.Scatter(
                    x=forecast_df['datetime'],
                    y=forecast_df['wind_speed'],
                    mode='lines+markers',
                    name='Wind Speed (m/s)',
                    line=dict(color='green'),
                    yaxis='y2'
                ))
                
                fig_conditions.update_layout(
                    title='Humidity & Wind Speed Forecast',
                    xaxis_title='Date & Time',
                    yaxis=dict(title='Humidity (%)', side='left'),
                    yaxis2=dict(title='Wind Speed (m/s)', side='right', overlaying='y'),
                    legend=dict(x=0, y=1),
                    height=400
                )
                
                st.plotly_chart(fig_conditions, use_container_width=True)
                
                # Daily power prediction forecast
                st.markdown("### ⚡ Predicted Power Output Forecast")
                
                # Calculate power predictions for forecast data
                forecast_predictions = []
                for _, row in forecast_df.iterrows():
                    # Calculate module temperature
                    module_temp_forecast = calculate_module_temperature(
                        row['temperature'], 
                        row['estimated_irradiation'], 
                        row['wind_speed']
                    )
                    
                    # Prepare prediction input
                    forecast_input = pd.DataFrame({
                        'AMBIENT_TEMPERATURE': [row['temperature']],
                        'MODULE_TEMPERATURE': [module_temp_forecast],
                        'IRRADIATION': [row['estimated_irradiation']],
                        'HUMIDITY': [row['humidity']],
                        'WIND_SPEED': [row['wind_speed']],
                        'HOUR': [row['datetime'].hour],
                        'MONTH': [row['datetime'].month],
                        'DAY_OF_YEAR': [row['datetime'].timetuple().tm_yday],
                        'TEMP_IRRADIATION': [row['temperature'] * row['estimated_irradiation']],
                        'IS_DAYTIME': [1 if 6 <= row['datetime'].hour <= 18 else 0]
                    })
                    
                    # Make prediction
                    prediction = model.predict(forecast_input)[0]
                    geo_adjusted_prediction = prediction * geo_info['suitability_score']
                    
                    forecast_predictions.append({
                        'datetime': row['datetime'],
                        'predicted_power': prediction,
                        'geo_adjusted_power': geo_adjusted_prediction,
                        'weather_condition': row['weather_main']
                    })
                
                # Convert to DataFrame and plot
                pred_df = pd.DataFrame(forecast_predictions)
                
                fig_power_forecast = go.Figure()
                
                fig_power_forecast.add_trace(go.Scatter(
                    x=pred_df['datetime'],
                    y=pred_df['predicted_power'],
                    mode='lines+markers',
                    name='Predicted Power (kW)',
                    line=dict(color='purple')
                ))
                
                fig_power_forecast.add_trace(go.Scatter(
                    x=pred_df['datetime'],
                    y=pred_df['geo_adjusted_power'],
                    mode='lines+markers',
                    name='Geo-Adjusted Power (kW)',
                    line=dict(color='darkgreen')
                ))
                
                fig_power_forecast.update_layout(
                    title='Solar Power Output Forecast',
                    xaxis_title='Date & Time',
                    yaxis_title='Power Output (kW)',
                    legend=dict(x=0, y=1),
                    height=400
                )
                
                st.plotly_chart(fig_power_forecast, use_container_width=True)
        
        with col2:
            # Current weather summary
            st.markdown("### 🌤️ Current Conditions")
            
            # Weather condition indicator
            weather_condition = weather_data['weather_main']
            if weather_condition == 'Clear':
                condition_color = 'green'
                condition_icon = '☀️'
            elif weather_condition == 'Clouds':
                condition_color = 'orange'
                condition_icon = '⛅'
            elif weather_condition in ['Rain', 'Drizzle']:
                condition_color = 'blue'
                condition_icon = '🌧️'
            else:
                condition_color = 'gray'
                condition_icon = '🌤️'
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {condition_color}20; border-radius: 10px;">
                <h2>{condition_icon}</h2>
                <h3>{weather_condition}</h3>
                <p>{weather_data['weather_description'].title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key weather metrics
            st.markdown("### 📊 Key Metrics")
            st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
            st.metric("Humidity", f"{weather_data['humidity']:.0f}%")
            st.metric("Wind Speed", f"{weather_data['wind_speed']:.1f} m/s")
            st.metric("Cloud Cover", f"{weather_data['clouds']:.0f}%")
            st.metric("Visibility", f"{weather_data['visibility']:.1f} km")
            
            # Solar potential indicator
            solar_potential = weather_data['estimated_irradiation'] / 1000
            st.markdown("### ☀️ Solar Potential")
            if solar_potential > 0.7:
                st.success(f"Excellent: {solar_potential:.1%}")
            elif solar_potential > 0.5:
                st.info(f"Good: {solar_potential:.1%}")
            else:
                st.warning(f"Limited: {solar_potential:.1%}")
            
            # Sunrise/sunset times
            st.markdown("### 🌅 Sun Times")
            st.markdown(f"**Sunrise:** {weather_data['sunrise'].strftime('%H:%M')}")
            st.markdown(f"**Sunset:** {weather_data['sunset'].strftime('%H:%M')}")
            
            # Data source info
            st.markdown("### ℹ️ Data Source")
            st.markdown(f"**Source:** {weather_data['data_source']}")
            st.markdown(f"**Last Updated:** {weather_data['last_updated'].strftime('%H:%M:%S')}")
            
            if weather_data['data_source'] == 'Simulated Data':
                st.info("💡 Enable real weather data in sidebar for live conditions.")
    
    with tab3:
        st.markdown('<h3 class="sub-header">🗺️ Real-Time Geological Maps</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create and display geological map
            geological_map = create_geological_map(latitude, longitude)
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            map_data = st_folium(geological_map, width=700, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display clicked location info
            if map_data['last_clicked']:
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lon = map_data['last_clicked']['lng']
                
                clicked_geo_info = get_geological_info(clicked_lat, clicked_lon)
                
                st.markdown("### 📍 Selected Location Analysis")
                st.markdown(f"**Coordinates:** {clicked_lat:.6f}, {clicked_lon:.6f}")
                st.markdown(f"**Geology Type:** {clicked_geo_info['geology_type']}")
                st.markdown(f"**Site Suitability Score:** {clicked_geo_info['suitability_score']:.2f}")
        
        with col2:
            st.markdown("### 🔍 Geological Features")
            
            # Display geological legend and information
            st.markdown("""
            **Rock Types:**
            - 🟫 Sedimentary: Good drainage, stable foundation
            - 🟥 Igneous: Very stable, excellent for heavy installations
            - 🟨 Metamorphic: Moderate stability, good drainage
            
            **Soil Types:**
            - 🟩 Alluvial: Excellent for solar installations
            - 🟧 Sandy: Good drainage, moderate stability
            - 🟦 Clay: Poor drainage, requires special foundations
            """)
            
            # Site selection recommendations
            st.markdown("### 💡 Site Selection Tips")
            st.markdown("""
            **Optimal Conditions:**
            - Stable geological formations
            - Good drainage (slope >2%)
            - Minimal earthquake risk
            - Easy access for maintenance
            
            **Avoid:**
            - Flood-prone areas
            - Unstable soil conditions
            - Areas with high seismic activity
            - Steep slopes (>20%)
            """)
            
            # Environmental considerations
            st.markdown("### 🌱 Environmental Impact")
            st.markdown("""
            **Positive Impacts:**
            - Reduced carbon footprint
            - Minimal land disturbance
            - Renewable energy generation
            
            **Mitigation Measures:**
            - Proper soil conservation
            - Wildlife corridor preservation
            - Sustainable construction practices
            """)
    
    with tab4:
        st.markdown('<h3 class="sub-header">📊 Detailed Analysis & Insights</h3>', 
                    unsafe_allow_html=True)
        
        # Sub-tabs for different analyses
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📈 Performance Analysis", "🔍 Feature Importance", "🌍 Geological Impact"])
        
        with analysis_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction vs Actual scatter plot
                fig_scatter = px.scatter(
                    x=y_test, 
                    y=y_pred,
                    labels={'x': 'Actual Power (kW)', 'y': 'Predicted Power (kW)'},
                    title="Predicted vs Actual Power Output"
                )
                fig_scatter.add_shape(
                    type="line",
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max(),
                    line=dict(color="red", dash="dash")
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Residuals histogram
                residuals = y_test - y_pred
                fig_residuals = px.histogram(
                    residuals,
                    nbins=30,
                    title="Prediction Residuals Distribution",
                    labels={'value': 'Residuals (kW)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Monthly performance analysis
            monthly_performance = df.groupby('MONTH').agg({
                'DC_POWER': ['mean', 'std', 'max'],
                'IRRADIATION': 'mean',
                'AMBIENT_TEMPERATURE': 'mean'
            }).round(2)
            
            st.markdown("### 📅 Monthly Performance Summary")
            st.dataframe(monthly_performance, use_container_width=True)
        
        with analysis_tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Solar Power Prediction"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                # Feature correlation heatmap
                correlation_matrix = df[features + ['DC_POWER']].corr()
                fig_heatmap = px.imshow(
                    correlation_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with analysis_tab3:
            # Geological impact analysis
            st.markdown("### 🌍 Geological Factors Impact on Solar Installation")
            
            # Create synthetic geological impact data
            geological_impact_data = pd.DataFrame({
                'Geology_Type': ['Sedimentary', 'Igneous', 'Metamorphic', 'Alluvial', 'Sandy', 'Clay'],
                'Installation_Cost_Factor': [1.0, 1.1, 1.05, 0.9, 0.95, 1.3],
                'Maintenance_Factor': [1.0, 0.9, 1.0, 0.8, 1.1, 1.4],
                'Lifespan_Factor': [1.0, 1.1, 1.05, 0.95, 0.9, 0.85],
                'Average_Suitability': [0.8, 0.85, 0.82, 0.9, 0.75, 0.65]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Installation cost by geology type
                fig_cost = px.bar(
                    geological_impact_data,
                    x='Geology_Type',
                    y='Installation_Cost_Factor',
                    title="Installation Cost Factor by Geology Type"
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            
            with col2:
                # Suitability by geology type
                fig_suitability = px.bar(
                    geological_impact_data,
                    x='Geology_Type',
                    y='Average_Suitability',
                    title="Average Site Suitability by Geology Type",
                    color='Average_Suitability',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_suitability, use_container_width=True)
            
            # Recommendations table
            st.markdown("### 📋 Geological Recommendations")
            recommendations = pd.DataFrame({
                'Geology Type': ['Alluvial Soil', 'Igneous Rock', 'Sedimentary Rock', 'Metamorphic Rock', 'Sandy Soil', 'Clay Soil'],
                'Recommendation': ['Excellent choice', 'Very good, stable', 'Good overall', 'Good with precautions', 'Moderate, monitor drainage', 'Avoid or use special foundations'],
                'Key Considerations': [
                    'Optimal drainage and stability',
                    'Excellent stability, may need specialized equipment',
                    'Good balance of stability and workability',
                    'Generally stable, check for fractures',
                    'Ensure proper drainage systems',
                    'Requires extensive foundation work'
                ]
            })
            st.dataframe(recommendations, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Solar Energy Production Prediction System with Geological Assessment | Supporting SDG 7: Affordable and Clean Energy</p>
        <p>Built with Streamlit, Folium, and Advanced Geospatial Analysis | Data Science for Sustainable Development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()