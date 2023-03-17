import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import folium
from streamlit_folium import folium_static

header = st.container()
countymap = st.container()
results = st.container()
options = st.sidebar
 
data_path = 'https://github.com/fatemey/real-estate-roi/blob/main/Data/'
with header:
    st.header('Prediction of Return on Investment (ROI) Across the US')
    st.write('**Please choose a county from the left sidebar or simply hover the pointer over any county on the map below.**')

with options:
    option_state = st.text_input(
        "Enter the US state",
        label_visibility='visible',
        placeholder='Maryland',
    )

    option_county = st.text_input(
        "Enter the county",
        label_visibility='visible',
        placeholder='Montgomery',
    )

    month = st.selectbox(
	    'Select the number of months for ROI',
	    [6, 9, 12]
	)

with countymap:
	# load data
	prediction = pd.read_parquet(data_path + 'prediction_' + str(month)).reset_index()
	data = pd.read_csv(data_path + 'RDC_Inventory_Core_Metrics_County_History.csv', dtype = {'county_fips': str})
	data = data.drop(len(data)-1)
	data.fillna({'county_name':'valdez-cordova, ak'}, inplace=True)
	df = pd.merge(prediction, data[['county_name','county_fips']], on='county_name', how='left')
	df = df.drop_duplicates().reset_index(drop=True)
	df['county_fips'] = df['county_fips'].apply(lambda x: str(x) if len(str(x))==5 else '0'+str(x) if len(str(x))==4 else '00'+str(x))
	with open(data_path + 'uscounties_20m.json') as f:
	    counties_data = json.load(f)
	gdf = gpd.read_file(data_path + "cb_2018_us_county_20m.shp")
	gdf['county_fips'] = gdf['STATEFP'] + gdf['COUNTYFP']
	dd = gdf.merge(df, on='county_fips', how='right')
	dd['investment'] = dd['y_pred'].apply(lambda x: 'Recommended' if x == 1 else 'Not Recommended')
	dd['county'] = dd['county_name'].apply(lambda x: str(x).split(',')[0].title())
	dd['state'] = dd['county_name'].apply(lambda x: str(x).split(',')[1].strip().upper())

	# map
	m = folium.Map(location=[40, -96], zoom_start=4)
	folium.Choropleth(
		geo_data=counties_data,
		name="County Data",
		data=df,
		columns=["county_fips", "y_pred"],
		key_on="properties.county_fips",
		fill_color="YlOrRd",
		bins=3,
		fill_opacity=0.8,
		line_opacity=.1,
		highlight=True,
		nan_fill_color="White",
		legend_name="Yellow: Investments Not Recommended, Red: Investments Recommended",
	).add_to(m)
	folium.features.GeoJson(
							data=dd,
							name='Investment Recommendations',
							style_function=lambda x: {'color':'black','fillColor':'transparent','weight':0.5},
							tooltip=folium.features.GeoJsonTooltip(
							    fields=['county',
							            'state',
							            'investment'],
							    aliases=["County Name:",
							    		 "State Name:",
							             "Investment:"], 
							    ),
							highlight_function=lambda x: {'weight':2,'fillColor':'black'}
							).add_to(m)
	folium.LayerControl().add_to(m)
	folium_static(m)

with results:
	prediction = pd.read_parquet(data_path + 'prediction_' + str(month))
	with open(data_path + 'state_abbrev.txt') as f:
		state_data = f.read()
	state_abbrev = json.loads(state_data)

	if option_state and option_county:
		state = option_state.lower().strip()
		county = option_county.lower().strip()

		if state in ['d.c.', 'dc', 'd. c.', 'district of columbia'] and county in ['d.c.', 'dc', 'd. c.', 'district of columbia']:
			result = int(prediction.loc['district of columbia, dc'])
			st.write(f'The expected ROI for the **District of Columbia** over the next **{month}** months is projected to be:')
		else:
			if state in str(state_abbrev.keys()).lower():
				abb = state_abbrev[state.capitalize()].lower()
			else:
				st.error('Please enter a valid state name')

			idx = county + ', ' + abb
			if idx in prediction.index:
				result = int(prediction.loc[idx])
			else:
				st.error('Please enter a valid county name') 
			
			if state in str(state_abbrev.keys()).lower() and idx in prediction.index:
				st.write(f'The expected ROI for the **{county.title()}** county in **{state.title()}** over the next **{month}** months is expected to be:')
		
		if result == 1:
			st.write('**Greater** than 5%!')
			st.write('Investment **is** recommended!')
		else:
			st.write('**Lower** than 5%!')
			st.write('Investment **is not** recommended!')
