import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# Most recent correct app, still need to make appearance pretty, find more statistics
# Maybe covid, wealth
st.set_page_config(layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 'intro'

STATE_BOUNDS = {
    "Alabama": {"lat_range": [30.1, 35.0], "lon_range": [-88.5, -84.9]},
    "Arizona": {"lat_range": [31.3, 37.0], "lon_range": [-114.8, -109.0]},
    "Arkansas": {"lat_range": [33.0, 36.5], "lon_range": [-94.6, -89.6]},
    "California": {"lat_range": [32.5, 42.0], "lon_range": [-124.5, -114.1]},
    "Colorado": {"lat_range": [36.9, 41.0], "lon_range": [-109.1, -102.0]},
    "Connecticut": {"lat_range": [40.9, 42.1], "lon_range": [-73.7, -71.8]},
    "Delaware": {"lat_range": [38.5, 39.8], "lon_range": [-75.8, -75.0]},
    "Florida": {"lat_range": [24.4, 31.0], "lon_range": [-87.6, -80.0]},
    "Georgia": {"lat_range": [30.4, 35.0], "lon_range": [-85.6, -80.8]},
    "Idaho": {"lat_range": [42.0, 49.0], "lon_range": [-117.2, -111.0]},
    "Illinois": {"lat_range": [36.9, 42.5], "lon_range": [-91.5, -87.5]},
    "Indiana": {"lat_range": [37.8, 41.8], "lon_range": [-88.1, -84.8]},
    "Iowa": {"lat_range": [40.4, 43.5], "lon_range": [-96.6, -90.1]},
    "Kansas": {"lat_range": [36.9, 40.0], "lon_range": [-102.1, -94.6]},
    "Kentucky": {"lat_range": [36.5, 39.1], "lon_range": [-89.6, -81.9]},
    "Louisiana": {"lat_range": [28.9, 33.0], "lon_range": [-94.0, -88.8]},
    "Maine": {"lat_range": [43.0, 47.5], "lon_range": [-71.1, -66.9]},
    "Maryland": {"lat_range": [37.9, 39.7], "lon_range": [-79.5, -75.0]},
    "Massachusetts": {"lat_range": [41.2, 42.9], "lon_range": [-73.5, -69.9]},
    "Michigan": {"lat_range": [41.7, 48.3], "lon_range": [-90.4, -82.4]},
    "Minnesota": {"lat_range": [43.5, 49.4], "lon_range": [-97.2, -89.5]},
    "Mississippi": {"lat_range": [30.2, 35.0], "lon_range": [-91.6, -88.0]},
    "Missouri": {"lat_range": [35.9, 40.6], "lon_range": [-95.8, -89.1]},
    "Montana": {"lat_range": [44.4, 49.0], "lon_range": [-116.1, -104.0]},
    "Nebraska": {"lat_range": [39.8, 43.0], "lon_range": [-104.1, -95.3]},
    "Nevada": {"lat_range": [35.0, 42.0], "lon_range": [-120.0, -114.0]},
    "New Hampshire": {"lat_range": [42.7, 45.3], "lon_range": [-72.6, -70.7]},
    "New Jersey": {"lat_range": [38.9, 41.4], "lon_range": [-75.6, -73.9]},
    "New Mexico": {"lat_range": [31.3, 37.0], "lon_range": [-109.0, -103.0]},
    "New York": {"lat_range": [40.5, 45.1], "lon_range": [-79.8, -71.9]},
    "North Carolina": {"lat_range": [33.8, 36.6], "lon_range": [-84.3, -75.5]},
    "North Dakota": {"lat_range": [45.9, 49.0], "lon_range": [-104.1, -96.6]},
    "Ohio": {"lat_range": [38.4, 41.9], "lon_range": [-84.8, -80.5]},
    "Oklahoma": {"lat_range": [33.6, 37.0], "lon_range": [-103.0, -94.4]},
    "Oregon": {"lat_range": [41.9, 46.3], "lon_range": [-124.6, -116.5]},
    "Pennsylvania": {"lat_range": [39.7, 42.3], "lon_range": [-80.5, -74.7]},
    "Rhode Island": {"lat_range": [41.1, 42.0], "lon_range": [-71.9, -71.1]},
    "South Carolina": {"lat_range": [32.0, 35.2], "lon_range": [-83.4, -78.5]},
    "South Dakota": {"lat_range": [42.4, 45.9], "lon_range": [-104.1, -96.4]},
    "Tennessee": {"lat_range": [34.9, 36.7], "lon_range": [-90.4, -81.6]},
    "Texas": {"lat_range": [25.8, 36.5], "lon_range": [-106.6, -93.5]},
    "Utah": {"lat_range": [36.9, 42.0], "lon_range": [-114.1, -109.0]},
    "Vermont": {"lat_range": [42.7, 45.0], "lon_range": [-73.5, -71.5]},
    "Virginia": {"lat_range": [36.5, 39.5], "lon_range": [-83.7, -75.2]},
    "Washington": {"lat_range": [45.5, 49.0], "lon_range": [-124.8, -116.9]},
    "West Virginia": {"lat_range": [37.2, 40.6], "lon_range": [-82.7, -77.7]},
    "Wisconsin": {"lat_range": [42.5, 47.1], "lon_range": [-92.9, -86.2]},
    "Wyoming": {"lat_range": [40.9, 45.0], "lon_range": [-111.0, -104.0]}
}

STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}


US_REGIONS = {
    "New England": ["Maine", "Rhode Island", "Connecticut", "Vermont", "New Hampshire", "Massachusetts"],
    "Mid-Atlantic": ["New York", "New Jersey", "Pennsylvania"],
    "Southeast": ["Virginia", "West Virginia", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama",
                  "Mississippi", "Tennessee", "Kentucky"],
    "Midwest": ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri",
                "North Dakota", "South Dakota", "Nebraska", "Kansas"],
    "Southwest": ["Texas", "Oklahoma", "New Mexico", "Arizona"],
    "West": ["California", "Nevada", "Utah", "Colorado", "Wyoming", "Montana", "Idaho",
             "Washington", "Oregon", "Alaska", "Hawaii"]
}


# Load Data Function
def data_load(data):
    try:
        return pd.read_csv(data)
    except FileNotFoundError:
        st.error(f"File '{data}' not found.")
        return pd.DataFrame()  # Return empty DataFrame on error


def intro_page():
    st.title("Welcome to MapMaster: Uncover U.S. Stats at a Glance!")
    st.write("""
        This app allows you to explore various statistical data related to population, housing, covid, employment data,
        and crime across different states and regions of the United States. 
        You can visualize the data on a map or view detailed tables of the data.
        To use this app, first you select either United States, or individual regions/states. 
        If you select the entire United States, you will be able to select two statistics per state 
        and see their correlation. 
        If you select individual states or regions as a whole, you will see statistics by the city level or aggregate to
        the county level. 
    """)
    if st.button("Start Exploring!"):
        st.session_state.page = 'map'

    # Create a layout with three columns
    col1, col2, col3 = st.columns([1.5, 2, 1])  # Adjust the weights as needed

    # Center the image in the middle column
    with col2:
        st.image("Headshot.png", width=300,
                 caption="Hello! My name is Connor Lewis. "
                         "I am a Senior at West Virginia University majoring in"
                         " Data Science with a minor in Business Data Analytics. "
                         "This project was made for DSCI 450 and is designed to be an interactive app.")


# Main Function
def map_page():
    # Load datasets
    # Load crime data and clean it
    crime_data_raw = data_load('total_crime.csv')
    # Replacing weird characters with no spaces
    crime_data_raw['states'].replace("\xa0", '', inplace=True, regex=True)
    crime_data_raw['states'].replace("8", '', inplace=True, regex=True)
    # Making cities two columns, city and county
    crime_data_raw[['city', 'county']] = crime_data_raw['cities'].str.split(', ', expand=True)
    crime_data_raw.drop(columns=['cities'])

    for column in ["population", "violent_crime", "murder", "rape", "robbery", "agrv_assault",  "prop_crime",
                   "burglary", "larceny", "vehicle_theft", "total_crime", "tot_violent_crime",
                   "tot_prop_crim", "arson"]:
        crime_data_raw[column] = pd.to_numeric(crime_data_raw[column], errors="coerce")

    crime_data_state = crime_data_raw.groupby('states',
                                              as_index=False).agg({'violent_crime': 'sum', 'murder': 'sum',
                                                                   'robbery': 'sum', 'agrv_assault': 'sum',
                                                                   'burglary': 'sum', 'larceny': 'sum',
                                                                   'vehicle_theft': 'sum', 'total_crime': 'sum',
                                                                   'tot_violent_crime': 'sum', 'arson': 'sum'})
    crime_data_state['state_abrev'] = crime_data_state['states'].replace(STATE_ABBREVIATIONS)
    # Load more data
    county_housing = data_load('housing_data_city.csv')
    state_housing = data_load('Housing_data_state.csv')
    df_pop = data_load('USPopulations.csv')
    df_cities = data_load("uscities.csv")

    # Clean and preprocess city housing data
    county_housing[['County', 'State_abbrev']] = county_housing['county_name'].str.split(',', expand=True)
    county_housing['County'] = county_housing['County'].str.strip()
    county_housing['State_abbrev'] = county_housing['State_abbrev'].str.strip().str.capitalize()

    # Clean population data
    df_pop['CITY'] = df_pop['CITY'].str.replace(' town', '').str.replace(' city', '')
    df_cities = df_cities[(df_cities['state_name'] != 'Hawaii') & (df_cities['state_name'] != 'Alaska')]

    # Merge population and city data
    df = pd.merge(df_pop, df_cities, left_on=['CITY', 'STATE'], right_on=['city', 'state_name'], how='left')
    df.drop(columns=['state_name', 'city', 'lat', 'lng'], inplace=True)
    df.rename(columns={
        "LONG": "lon",
        "LAT": "lat",
        "2022_POPULATION": "Population",
        "CITY": "city",
        "STATE": "state_name",
        'state_id': "state_abbreviation"
    }, inplace=True)

    st.title("MapMaster: Uncover U.S. Statistics at a Glance")

    # Region selection
    region_options = ["United States", "All States"] + list(US_REGIONS.keys()) + df['state_name'].unique().tolist()
    selected_region_or_states = st.multiselect("Select United States, All States, Regions, or Individual States",
                                               region_options, default=["United States"])

    selected_states = set()
    for selection in selected_region_or_states:
        if selection in US_REGIONS:
            selected_states.update(US_REGIONS[selection])
        elif selection in df['state_name'].unique():
            selected_states.add(selection)

    selected_states = list(selected_states)

    # Map view for United States
    if "United States" in selected_region_or_states and len(selected_region_or_states) == 1:
        state_data = df.groupby('state_abbreviation', as_index=False)['population'].sum()
        state_data = pd.merge(state_data, state_housing, left_on='state_abbreviation', right_on='state_id', how='inner')
        state_data = pd.merge(state_data, crime_data_state, left_on='state_abbreviation', right_on='state_abrev',
                              how='left')
        state_data.drop(columns=['month_date_yyyymm'], inplace=True)

        state_data.rename(columns={"population": "Population Totals",
                                   "median_listing_price": "Median House Listing Price",
                                   "active_listing_count": "Active House Listings",
                                   "median_days_on_market": "Median Days House on Market",
                                   "new_listing_count": "New Housing Listings",
                                   "price_increased_count": "Housing Price Increases",
                                   "price_reduced_count": "Housing Price Reductions",
                                   "pending_listing_count": "Pending Housing Listings",
                                   'median_listing_price_per_square_foot': "Median House Listing Price per Square Foot",
                                   'median_square_feet': "Median House Square Footage",
                                   'average_listing_price': 'Average House Listing Price',
                                   'total_listing_count': "Total House Listings",
                                   'violent_crime': "Violent Crime Totals per 100k People",
                                   'murder': "Murder Totals per 100k People",
                                   'robbery': 'Robbery Totals per 100k People',
                                   'agrv_assault': 'Aggravated Assault Totals per 100k People',
                                   'burglary': "Burglary Totals per 100k People",
                                   'larceny': "Larceny Totals per 100k People",
                                   'vehicle_theft': 'Vehicle Theft Totals per 100k People',
                                   'total_crime': 'Total Crime per 100k People',
                                   'tot_violent_crime': 'Total Violent Crime per 100k People',
                                   'arson': 'Arson Total per 100k People'
                                   }, inplace=True)
        # Setting the statistics to be per 100k people for each of the crime columns
        state_data["Violent Crime Totals per 100k People"] = (
                (state_data['Violent Crime Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Murder Totals per 100k People'] = (
                (state_data['Murder Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Robbery Totals per 100k People'] = (
                (state_data['Robbery Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Aggravated Assault Totals per 100k People'] = (
                (state_data['Aggravated Assault Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Burglary Totals per 100k People'] = (
                (state_data['Burglary Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Larceny Totals per 100k People'] = (
                (state_data['Larceny Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Vehicle Theft Totals per 100k People'] = (
                (state_data['Vehicle Theft Totals per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Total Crime per 100k People'] = (
                (state_data['Total Crime per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Total Violent Crime per 100k People'] = (
                (state_data['Total Violent Crime per 100k People']/state_data['Population Totals']) * 100000)

        state_data['Arson Total per 100k People'] = (
                (state_data['Arson Total per 100k People']/state_data['Population Totals']) * 100000)

        # Divider
        tab1, tab2 = st.tabs(['Map View', 'Table View'])

        with tab1:
            stat_options = [col for col in state_data.columns if col not in ['state_abbreviation',
                                                                             'month_date_yyyymm',
                                                                             'state',
                                                                             'state_id',
                                                                             'state_abrev']]
            stat1 = st.selectbox('Select Statistic for Left Map', stat_options)
            stat2 = st.selectbox('Select Statistic for Right Map', stat_options)

            col1, col2 = st.columns(2)

            # Correlation plot
            # Filter out NaN values for selected statistics
            # Check if the same statistic is selected for both stat1 and stat2
            if stat1 == stat2:
                st.warning(
                    f"You've selected the same statistic ({stat1}) "
                    f"for both axes. The correlation is trivially perfect (1.0).")
            else:
                # Filter out NaN values for selected statistics
                filtered_data = state_data[['state_abbreviation', stat1, stat2]].dropna()

                # Correlation plot
                fig_corr = px.scatter(
                    filtered_data,
                    x=stat1,
                    y=stat2,
                    hover_data=['state_abbreviation'],
                    labels={stat1: stat1.capitalize(), stat2: stat2.capitalize()},
                    title=f"Correlation between {stat1.capitalize()} and {stat2.capitalize()}",
                )

                # Calculate the line of best fit
                slope, intercept = np.polyfit(filtered_data[stat1], filtered_data[stat2], 1)
                line_of_best_fit = slope * filtered_data[stat1] + intercept

                # Add the best fit line to the scatter plot
                fig_corr.add_trace(go.Scatter(
                    x=filtered_data[stat1],
                    y=line_of_best_fit,
                    mode='lines',
                    name='Best Fit Line',
                    line=dict(color='red')
                ))

                # Calculate the correlation coefficient ignoring NaN values
                corr_coef = np.corrcoef(filtered_data[stat1], filtered_data[stat2])[0, 1]

                # Display the plot and the correlation coefficient
                st.plotly_chart(fig_corr, use_container_width=True)
                st.write(
                    f"Correlation coefficient between {stat1.capitalize()} and {stat2.capitalize()}: {corr_coef:.2f}")
                # Left map
                with col1:
                    fig1 = px.choropleth(
                        state_data,
                        locations='state_abbreviation',
                        locationmode='USA-states',
                        color=stat1,
                        hover_name='state_abbreviation',
                        hover_data=[stat1],
                        scope='usa',
                        color_continuous_scale="Bluered",
                        labels={stat1: stat1},
                        title=f"State-level {stat1.capitalize()} For Contiguous United States"
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # Right map
                with col2:
                    fig2 = px.choropleth(
                        state_data,
                        locations='state_abbreviation',
                        locationmode='USA-states',
                        color=stat2,
                        hover_name='state_abbreviation',
                        hover_data=[stat2],
                        scope='usa',
                        color_continuous_scale="Bluered",
                        labels={stat2: stat2},
                        title=f"State-level {stat2.capitalize()} For Contiguous United States"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            columns_to_exclude = []
            state_data_table = state_data.drop(columns=columns_to_exclude)
            st.dataframe(state_data_table)

    # Map view for All States
    elif "All States" in selected_region_or_states:
        city_data = df.copy()
        city_data['Log Population'] = np.log10(city_data['Population'])

        city_data = pd.merge(city_data, crime_data_raw, left_on=['city', 'state_name'],
                             right_on=['city', 'states'], how='left')
        tab1, tab2 = st.tabs(['Map View', 'Table View'])

        with tab1:
            min_population, max_population = st.slider(
                "Select Population Range",
                min_value=int(city_data['Population'].min()),
                max_value=int(city_data['Population'].max()),
                value=(int(city_data['Population'].min()), int(city_data['Population'].max()))
            )

            city_data = city_data[(city_data['Population'] >= min_population) &
                                  (city_data['Population'] <= max_population)]
            city_data.drop(columns=['rape'], inplace=True)
            city_data.rename(columns={'violent_crime': "Violent Crime Totals per 100k People",
                                      'murder': "Murder Totals per 100k People",
                                      'robbery': 'Robbery Totals per 100k People',
                                      'agrv_assault': 'Aggravated Assault Totals per 100k People',
                                      'burglary': "Burglary Totals per 100k People",
                                      'larceny': "Larceny Totals per 100k People",
                                      'vehicle_theft': 'Vehicle Theft Totals per 100k People',
                                      'total_crime': 'Total Crime per 100k People',
                                      'tot_violent_crime': 'Total Violent Crime per 100k People',
                                      'arson': 'Arson Total per 100k People'}, inplace=True)
            city_data["Violent Crime Totals per 100k People"] = (
                    (city_data['Violent Crime Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Murder Totals per 100k People'] = (
                    (city_data['Murder Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Robbery Totals per 100k People'] = (
                    (city_data['Robbery Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Aggravated Assault Totals per 100k People'] = (
                    (city_data['Aggravated Assault Totals per 100k People'] / city_data[
                        'Population']) * 100000)

            city_data['Burglary Totals per 100k People'] = (
                    (city_data['Burglary Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Larceny Totals per 100k People'] = (
                    (city_data['Larceny Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Vehicle Theft Totals per 100k People'] = (
                    (city_data['Vehicle Theft Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Total Crime per 100k People'] = (
                    (city_data['Total Crime per 100k People'] / city_data['Population']) * 100000)

            city_data['Total Violent Crime per 100k People'] = (
                    (city_data['Total Violent Crime per 100k People'] / city_data['Population']) * 100000)

            city_data['Arson Total per 100k People'] = (
                    (city_data['Arson Total per 100k People'] / city_data['Population']) * 100000)

            stat_options = city_data.select_dtypes(include=np.number).columns.tolist()  # Get numerical columns
            items_to_remove = ['density', 'lat', 'lon', 'county_fips', 'Unnamed: 0', 'population_y', 'id', 'ranking',
                               'population_x', 'prop_crime', 'tot_prop_crim']
            stat_options = [item for item in stat_options if item not in items_to_remove]
            color_by = st.selectbox("Select Column to Color Cities By", options=stat_options,
                                    index=stat_options.index('Log Population'))

            aggregate_by_county = st.checkbox("Aggregate by County", value=False)
            if aggregate_by_county:
                county_data = city_data.groupby(['county_name', 'state_name'], as_index=False).agg(
                    {'Population': 'sum'})
                county_data['state_abbreviation'] = county_data['state_name'].replace(STATE_ABBREVIATIONS)
                county_data['county_name'] = county_data['county_name'].str.lower()

                county_housing['State_abbrev'] = county_housing['State_abbrev'].str.upper()
                county_data = pd.merge(county_data, county_housing, left_on=['county_name', 'state_abbreviation'],
                                       right_on=['County', 'State_abbrev'], how='left')

                county_data.drop(columns=['county_name_x', 'month_date_yyyymm', 'county_name_y', 'state_abbreviation',
                                          'State_abbrev'],
                                 inplace=True)
                county_data.rename(columns={"county_fips": "County Fips Code",
                                            "state_name": "State Name",
                                            "median_listing_price": "Median House Listing Price",
                                            "active_listing_count": "Active House Listings",
                                            "median_days_on_market": "Median Days House on Market",
                                            "new_listing_count": "New Housing Listings",
                                            "price_increased_count": "Housing Price Increases",
                                            "price_reduced_count": "Housing Price Reductions",
                                            "pending_listing_count": "Pending Housing Listings",
                                            'median_listing_price_per_square_foot':
                                                "Median House Listing Price per Square Foot",
                                            'median_square_feet': "Median House Square Footage",
                                            'average_listing_price': 'Average House Listing Price',
                                            'total_listing_count': "Total House Listings",
                                            }, inplace=True)
                county_data['Log Population'] = np.log10(county_data['Population'])
                stat_options = county_data.select_dtypes(include=np.number).columns.tolist()
                color_by = st.selectbox("Select Column to Color Counties By", options=stat_options,
                                        index=stat_options.index('Log Population'))
                # SOME COUNTIES NOT SHOWING UP
                county_data = county_data.dropna(subset=['County Fips Code'])  # Drops rows with missing FIPS
                county_data['County Fips Code'] = county_data['County Fips Code'].apply(lambda x: f'{int(x):05d}')
                fig = px.choropleth(county_data,
                                    geojson="https://raw.githubusercontent.com"
                                            "/plotly/datasets/master/geojson-counties-fips.json",
                                    # GeoJSON for counties
                                    locations='County Fips Code',  # 'county_fips' column for geographical reference
                                    color=color_by,
                                    # Column to visualize (color by Median House Listing Price)
                                    hover_name='County',  # Column to show on hover
                                    hover_data={
                                        'Population': True,
                                        'Active House Listings': True,
                                        'Median Days House on Market': True,
                                        'New Housing Listings': True,
                                        'Housing Price Increases': True,
                                        'Housing Price Reductions': True,
                                        'Pending Housing Listings': True,
                                        'State Name': True
                                    },
                                    scope="usa",  # Limit map to USA
                                    title='Median House Listing Price by County'
                                    )

                # Update layout for better display
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

                # Show the figure
                st.plotly_chart(fig, use_container_width=True)
                # st.dataframe(county_data)

            else:
                city_data = city_data.dropna(subset=[color_by])
                fig = px.scatter_mapbox(
                    city_data,
                    lat='lat',
                    lon='lon',
                    hover_name='city',
                    hover_data=['state_name', 'Population'],
                    color=color_by,
                    color_continuous_scale="Bluered",
                    title=f"City-level Population in {', '.join(selected_states)}",
                    height=700
                )

                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_center={"lat": np.mean(city_data['lat']), "lon": np.mean(city_data['lon'])},
                    width=1200,
                    height=700,
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(city_data)

    # Map view for selected states
    else:
        tab1, tab2 = st.tabs(['Map View', 'Table View'])
        city_data = df[df['state_name'].isin(selected_states)]
        city_data['Log Population'] = np.log10(city_data['Population'])
        # Crime Combining
        city_data = pd.merge(city_data, crime_data_raw, left_on=['city', 'state_name'], right_on=['city', 'states'],
                             how='left')

        with tab1:
            min_population, max_population = st.slider(
                "Select Population Range",
                min_value=int(city_data['Population'].min()),
                max_value=int(city_data['Population'].max()),
                value=(int(city_data['Population'].min()), int(city_data['Population'].max()))
            )
            city_data = city_data[(city_data['Population'] >= min_population) &
                                  (city_data['Population'] <= max_population)]
            city_data.drop(columns=['rape'], inplace=True)
            city_data.rename(columns={'violent_crime': "Violent Crime Totals per 100k People",
                                      'murder': "Murder Totals per 100k People",
                                      'robbery': 'Robbery Totals per 100k People',
                                      'agrv_assault': 'Aggravated Assault Totals per 100k People',
                                      'burglary': "Burglary Totals per 100k People",
                                      'larceny': "Larceny Totals per 100k People",
                                      'vehicle_theft': 'Vehicle Theft Totals per 100k People',
                                      'total_crime': 'Total Crime per 100k People',
                                      'tot_violent_crime': 'Total Violent Crime per 100k People',
                                      'arson': 'Arson Total per 100k People'}, inplace=True)
            city_data["Violent Crime Totals per 100k People"] = (
                    (city_data['Violent Crime Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Murder Totals per 100k People'] = (
                    (city_data['Murder Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Robbery Totals per 100k People'] = (
                    (city_data['Robbery Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Aggravated Assault Totals per 100k People'] = (
                    (city_data['Aggravated Assault Totals per 100k People'] / city_data[
                        'Population']) * 100000)

            city_data['Burglary Totals per 100k People'] = (
                    (city_data['Burglary Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Larceny Totals per 100k People'] = (
                    (city_data['Larceny Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Vehicle Theft Totals per 100k People'] = (
                    (city_data['Vehicle Theft Totals per 100k People'] / city_data['Population']) * 100000)

            city_data['Total Crime per 100k People'] = (
                    (city_data['Total Crime per 100k People'] / city_data['Population']) * 100000)

            city_data['Total Violent Crime per 100k People'] = (
                    (city_data['Total Violent Crime per 100k People'] / city_data['Population']) * 100000)

            city_data['Arson Total per 100k People'] = (
                    (city_data['Arson Total per 100k People'] / city_data['Population']) * 100000)

            stat_options = city_data.select_dtypes(include=np.number).columns.tolist()  # Get numerical columns
            items_to_remove = ['density', 'lat', 'lon', 'county_fips', 'Unnamed: 0', 'population_y', 'id', 'ranking',
                               'population_x', 'prop_crime', 'tot_prop_crime']
            stat_options = [item for item in stat_options if item not in items_to_remove]
            color_by = st.selectbox("Select Column to Color By", options=stat_options,
                                    index=stat_options.index('Log Population'))
            aggregate_by_county = st.checkbox("Aggregate by County", value=False)
            if aggregate_by_county:
                county_data = city_data.groupby(['county_name', 'state_name'],
                                                as_index=False).agg({'Population': 'sum'})

                county_data['state_abbreviation'] = county_data['state_name'].replace(STATE_ABBREVIATIONS)
                county_data['county_name'] = county_data['county_name'].str.lower()

                county_housing['State_abbrev'] = county_housing['State_abbrev'].str.upper()

                county_data = pd.merge(county_data, county_housing, left_on=['county_name', 'state_abbreviation'],
                                       right_on=['County', 'State_abbrev'], how='left')
                county_data.drop(columns=['county_name_x', 'month_date_yyyymm', 'county_name_y', 'state_abbreviation',
                                          'State_abbrev'],
                                 inplace=True)
                county_data.rename(columns={"county_fips": "County Fips Code",
                                            "state_name": "State Name",
                                            "median_listing_price": "Median House Listing Price",
                                            "active_listing_count": "Active House Listings",
                                            "median_days_on_market": "Median Days House on Market",
                                            "new_listing_count": "New Housing Listings",
                                            "price_increased_count": "Housing Price Increases",
                                            "price_reduced_count": "Housing Price Reductions",
                                            "pending_listing_count": "Pending Housing Listings",
                                            'median_listing_price_per_square_foot':
                                                "Median House Listing Price per Square Foot",
                                            'median_square_feet': "Median House Square Footage",
                                            'average_listing_price': 'Average House Listing Price',
                                            'total_listing_count': "Total House Listings",
                                            }, inplace=True)

                # st.dataframe(county_data)

                # Create a choropleth map using Plotly
                county_data['Log Population'] = np.log10(county_data['Population'])
                stat_options = county_data.select_dtypes(include=np.number).columns.tolist()
                color_by = st.selectbox("Select Column to Color Counties By", options=stat_options,
                                        index=stat_options.index('Log Population'))
                county_data = county_data.dropna(subset=['County Fips Code'])  # Drops rows with missing FIPS
                county_data['County Fips Code'] = county_data['County Fips Code'].apply(lambda x: f'{int(x):05d}')

                fig = px.choropleth(county_data,
                                    geojson="https://raw.githubusercontent.com"
                                            "/plotly/datasets/master/geojson-counties-fips.json",
                                    # GeoJSON for counties
                                    locations='County Fips Code',  # 'county_fips' column for geographical reference
                                    color=color_by,
                                    # Column to visualize (color by Median House Listing Price)
                                    hover_name='County',  # Column to show on hover
                                    hover_data={
                                        'Population': True,
                                        'Active House Listings': True,
                                        'Median Days House on Market': True,
                                        'New Housing Listings': True,
                                        'Housing Price Increases': True,
                                        'Housing Price Reductions': True,
                                        'Pending Housing Listings': True,
                                    },
                                    scope="usa",  # Limit map to USA
                                    title='Median House Listing Price by County'
                                    )

                # Update layout for better display
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})

                # Show the figure
                st.plotly_chart(fig, use_container_width=True)

            else:
                city_data = city_data.dropna(subset=[color_by])
                fig = px.scatter_mapbox(
                    city_data,
                    lat='lat',
                    lon='lon',
                    hover_name='city',
                    hover_data=['state_name', 'Population'],
                    color=color_by,
                    color_continuous_scale="Bluered",
                    title=f"City-level Population in {', '.join(selected_states)}",
                    zoom=3,
                    height=700
                )

                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_center={"lat": np.mean(city_data['lat']), "lon": np.mean(city_data['lon'])},
                    width=1200,
                    height=700,
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(city_data)


def main():
    if st.session_state.page == 'intro':
        intro_page()
    elif st.session_state.page == 'map':
        map_page()


if __name__ == "__main__":
    main()
