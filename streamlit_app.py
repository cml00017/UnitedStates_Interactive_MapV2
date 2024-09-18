import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
# When getting state data, join it directly to the united states map section to avoid having to aggregate
# when getting city level data, will have to join directly on to cities and then aggregate from there
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
    "Southeast": ["Virginia", "West Virginia", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Mississippi", "Tennessee", "Kentucky"],
    "Midwest": ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri", "North Dakota", "South Dakota", "Nebraska", "Kansas"],
    "Southwest": ["Texas", "Oklahoma", "New Mexico", "Arizona"],
    "West": ["California", "Nevada", "Utah", "Colorado", "Wyoming", "Montana", "Idaho", "Washington", "Oregon", "Alaska", "Hawaii"]
}


# Define US regions
US_REGIONS = {
    "New England": ["Maine", "Rhode Island", "Connecticut", "Vermont", "New Hampshire", "Massachusetts"],
    "Mid-Atlantic": ["New York", "New Jersey", "Pennsylvania"],
    "Southeast": ["Virginia", "West Virginia", "North Carolina", "South Carolina", "Georgia", "Florida", "Alabama", "Mississippi", "Tennessee", "Kentucky"],
    "Midwest": ["Ohio", "Indiana", "Illinois", "Michigan", "Wisconsin", "Minnesota", "Iowa", "Missouri", "North Dakota", "South Dakota", "Nebraska", "Kansas"],
    "Southwest": ["Texas", "Oklahoma", "New Mexico", "Arizona"],
    "West": ["California", "Nevada", "Utah", "Colorado", "Wyoming", "Montana", "Idaho", "Washington", "Oregon", "Alaska", "Hawaii"]
}


# Load Data Function
def data_load(data):
    return pd.read_csv(data)


# Main Function
def main():
    city_housing = data_load('housing_data_city.csv')
    city_housing[['County', 'State_abbrev']] = city_housing['county_name'].str.split(',', expand=True)
    city_housing['County'] = city_housing['County'].str.strip()
    city_housing['State_abbrev'] = city_housing['State_abbrev'].str.strip()

    city_housing['State_abbrev'] = city_housing['State_abbrev'].str.capitalize()

    state_housing = data_load('Housing_data_state.csv')
    dfPop = data_load('USPopulations.csv')
    dfCities = data_load("uscities.csv")
    dfPop['CITY'] = dfPop['CITY'].replace({' town': '', ' city': ''}, regex=True)
    dfCities = dfCities[(dfCities['state_name'] != 'Hawaii') & (dfCities['state_name'] != 'Alaska')]

    df = pd.merge(dfPop, dfCities, left_on=['CITY', 'STATE'], right_on=['city', 'state_name'], how='left')
    df.drop(columns=['state_name', 'city', 'lat', 'lng'], inplace=True)
    # Rename columns for easier referencing
    df.rename(columns={
        "LONG": "lon",
        "LAT": "lat",
        "2022_POPULATION": "Population",
        "CITY": "city",
        "STATE": "state_name",
        'state_id': "state_abbreviation"
    }, inplace=True)
    st.title("Population Map Visualization")

    # Dropdown menu for selecting regions, states, or the entire US
    region_options = ["United States", "All States"] + list(US_REGIONS.keys()) + df['state_name'].unique().tolist()
    selected_region_or_states = st.multiselect("Select United States, All States, Regions, or Individual States", region_options, default=["United States"])

    # Expand selected regions to their corresponding states
    selected_states = set()
    for selection in selected_region_or_states:
        if selection in US_REGIONS:
            selected_states.update(US_REGIONS[selection])
        elif selection in df['state_name'].unique():
            selected_states.add(selection)

    selected_states = list(selected_states)

    # Handle the case for "United States" (aggregate by state)
    if "United States" in selected_region_or_states and len(selected_region_or_states) == 1:
        # Group by state and sum population
        state_data = df.groupby('state_abbreviation', as_index=False)['population'].sum()
        state_data = pd.merge(state_data, state_housing, left_on='state_abbreviation', right_on='state_id', how='inner')
        tab1, tab2 = st.tabs(['Map View', 'Table View'])
        with tab1:
            # Create a state-level choropleth map
            stat_options = list(state_data.columns)
            stat_options.remove('state_abbreviation')
            stat_options.remove('month_date_yyyymm')
            stat_options.remove('state')
            stat_options.remove('state_id')
            stat1 = st.selectbox('Select Statistic for Left Map', stat_options)
            stat2 = st.selectbox('Select Statistic for Right Map', stat_options)

            # Split View
            col1, col2 = st.columns(2)

            with col1:
                # Create the first map
                fig1 = px.choropleth(
                    state_data,
                    locations='state_abbreviation',
                    locationmode='USA-states',
                    color=stat1,
                    hover_name='state_abbreviation',
                    hover_data=[stat1],
                    scope='usa',
                    color_continuous_scale="Bluered",  # Blue to Red color scale
                    labels={stat1: stat1},
                    title=f"State-level {stat1.capitalize()} For Contiguous United States"
                )

                fig1.update_layout(
                    width=900,  # Set the width of the map
                    height=500,  # Set the height of the map
                )

                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Create the second map
                fig2 = px.choropleth(
                    state_data,
                    locations='state_abbreviation',
                    locationmode='USA-states',
                    color=stat2,
                    hover_name='state_abbreviation',
                    hover_data=[stat2],
                    scope='usa',
                    color_continuous_scale="Bluered",  # Blue to Red color scale
                    labels={stat2: stat2},
                    title=f"State-level {stat2.capitalize()} For Contiguous United States"
                )

                fig2.update_layout(
                    width=900,  # Set the width of the map
                    height=500,  # Set the height of the map
                )

                st.plotly_chart(fig2, use_container_width=True)
            # Correlation chart
            fig_corr = px.scatter(
                state_data,
                x=stat1,
                y=stat2,
                hover_data=['state_abbreviation'],
                labels={stat1: stat1.capitalize(), stat2: stat2.capitalize()},
                title=f"Correlation between {stat1.capitalize()} and {stat2.capitalize()}",
            )

            # Calculate line of best fit
            slope, intercept = np.polyfit(state_data[stat1], state_data[stat2], 1)
            line_of_best_fit = slope * state_data[stat1] + intercept

            # Add line of best fit to the scatter plot
            fig_corr.add_trace(go.Scatter(
                x=state_data[stat1],
                y=line_of_best_fit,
                mode='lines',
                name='Best Fit Line',
                line=dict(color='red')
            ))

            # Calculate correlation coefficient
            corr_coef = np.corrcoef(state_data[stat1], state_data[stat2])[0, 1]
            st.plotly_chart(fig_corr, use_container_width=True)
            st.write(f"Correlation coefficient between {stat1.capitalize()} and {stat2.capitalize()}: {corr_coef:.2f}")

        with tab2:
            st.dataframe(state_data)

    # Handle the case for "All States" (city-level data for all states)
    elif "All States" in selected_region_or_states:
        # Display city-level data for all states
        city_data = df.copy()
        city_data['city'].replace({' town': '', ' city': ''}, regex=True, inplace=True)
        city_data['log_population'] = np.log10(city_data['population'])

        tab1, tab2 = st.tabs(['Map View', 'Table View'])
        with tab1:
            # Population range filter
            min_population, max_population = st.slider(
                "Select Population Range",
                min_value=int(city_data['population'].min()),
                max_value=int(city_data['population'].max()),
                value=(int(city_data['population'].min()), int(city_data['population'].max()))
            )
            city_data = city_data[(city_data['population'] >= min_population) & (city_data['population'] <= max_population)]

            # Adjust bounds for all states
            lat_range = [min(city_data['lat']), max(city_data['lat'])]
            lon_range = [min(city_data['lon']), max(city_data['lon'])]

            # Create a city-level scatter plot map using Mapbox for street view
            fig = px.scatter_mapbox(
                city_data,
                lat='lat',
                lon='lon',
                hover_name='city',
                hover_data=['state_name', 'population'],
                color='log_population',
                color_continuous_scale="Bluered",  # Blue to Red color scale
                title="City-level Population for All States",
                zoom=3,
                height=700
            )

            # Set the mapbox style to "open-street-map" for default street view
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=3,
                mapbox_center={"lat": np.mean(lat_range), "lon": np.mean(lon_range)},
                width=1200,  # Set the width of the map
                height=700,  # Set the height of the map
            )

            # Display the city-level map
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(city_data[['city', 'state_name', 'population']])

    # Handle the case for individual states or selected regions
    else:
        # Filter the data to only the selected states
        tab1, tab2 = st.tabs(['Map View', 'Table View'])
        with tab1:
            city_data = df[df['state_name'].isin(selected_states)]
            city_data['city'].replace({' town': '', ' city': ''}, regex=True, inplace=True)
            city_data['log_population'] = np.log10(city_data['population'])

            min_population, max_population = st.slider(
                "Select Population Range",
                min_value=int(city_data['population'].min()),
                max_value=int(city_data['population'].max()),
                value=(int(city_data['population'].min()), int(city_data['population'].max()))
            )
            city_data = city_data[(city_data['population'] >= min_population) & (city_data['population'] <= max_population)]

            # Adjust bounds for multiple states
            lat_range = [min(city_data['lat']), max(city_data['lat'])]
            lon_range = [min(city_data['lon']), max(city_data['lon'])]

            # Create a city-level scatter plot map using Mapbox for street view
            fig = px.scatter_mapbox(
                city_data,
                lat='lat',
                lon='lon',
                hover_name='city',
                hover_data=['state_name', 'population'],
                color='log_population',
                color_continuous_scale="Bluered",  # Blue to Red color scale
                title=f"City-level Population in {', '.join(selected_states)}",
                zoom=5,
                height=700
            )

            # Set the mapbox style to "open-street-map"
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_zoom=5,
                mapbox_center={"lat": np.mean(lat_range), "lon": np.mean(lon_range)},
                width=1200,  # Set the width of the map
                height=700,  # Set the height of the map
            )

            # Display the city-level map
            # When clicking on city dot, go to 3d view
            # Also incorporate split map for statistics
            # Be able to select county aggregate or cities
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            #st.dataframe(city_data[['city', 'state_name', 'population']])
            st.dataframe(city_data)


if __name__ == "__main__":
    main()

