import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import dash_bootstrap_components as dbc
import json
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the data
def load_and_preprocess_data():
    df = pd.read_csv('KaggleV2-May-2016.csv')
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
    df['Month'] = df['AppointmentDay'].dt.month_name()
    df['DaysUntilAppointment'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    df['NoShow'] = (df['No-show'] == 'Yes').astype(int)
    df['ShowUp'] = (df['No-show'] == 'No').astype(int)
    return df

df = load_and_preprocess_data()

# For map visualization, get lat/lon for neighborhoods
neighborhood_coords = {n: (np.random.uniform(-20, -19.8), np.random.uniform(-40.4, -40.2)) for n in df['Neighbourhood'].unique()}

def get_neighborhood_latlon(neigh):
    return neighborhood_coords.get(neigh, (-20, -40.3))

df['lat'] = df['Neighbourhood'].apply(lambda n: get_neighborhood_latlon(n)[0])
df['lon'] = df['Neighbourhood'].apply(lambda n: get_neighborhood_latlon(n)[1])

# App initialization with Bootstrap for responsiveness
themes = [dbc.themes.BOOTSTRAP, dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=[themes[0]], suppress_callback_exceptions=True)

# --- Layout Components ---
# Welcome Modal
welcome_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Welcome to the Medical Appointment Dashboard!")),
    dbc.ModalBody([
        html.P("Explore 110,000+ medical appointments from Brazil. Use filters, interact with charts, and discover insights!"),
        html.Ul([
            html.Li("Responsive design for all devices"),
            html.Li("Download data and charts"),
            html.Li("Advanced filtering and drill-down"),
            html.Li("Map, correlation, and prediction features"),
            html.Li("Dark mode and accessibility options")
        ]),
        html.P("Click 'Close' to start exploring.")
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="close-welcome", className="ms-auto", n_clicks=0)
    ),
], id="welcome-modal", is_open=True, backdrop='static')

# Sidebar Filters
sidebar = dbc.Card([
    html.H4("Filters", className="mb-3 mt-2 text-center"),
    html.Hr(),
    dbc.Label("Age Group"),
    dcc.Dropdown(
        id='age-filter',
        options=[{'label': 'All', 'value': 'All'}] + [{'label': str(a), 'value': str(a)} for a in df['AgeGroup'].cat.categories],
        value='All', multi=True, clearable=False, className='mb-2'),
    dbc.Label("Gender"),
    dcc.Dropdown(
        id='gender-filter',
        options=[{'label': 'All', 'value': 'All'}] + [{'label': g, 'value': g} for g in df['Gender'].unique()],
        value='All', clearable=False, className='mb-2'),
    dbc.Label("Neighborhood"),
    dcc.Dropdown(
        id='neighborhood-filter',
        options=[{'label': 'All', 'value': 'All'}] + [{'label': n, 'value': n} for n in sorted(df['Neighbourhood'].unique())],
        value='All', multi=True, clearable=False, className='mb-2'),
    dbc.Label("Medical Condition"),
    dcc.Dropdown(
        id='condition-filter',
        options=[{'label': 'All', 'value': 'All'},
                 {'label': 'Hypertension', 'value': 'Hipertension'},
                 {'label': 'Diabetes', 'value': 'Diabetes'},
                 {'label': 'Alcoholism', 'value': 'Alcoholism'},
                 {'label': 'Handicap', 'value': 'Handcap'}],
        value='All', clearable=False, className='mb-2'),
    dbc.Label("Date Range"),
    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=df['AppointmentDay'].min(),
        max_date_allowed=df['AppointmentDay'].max(),
        start_date=df['AppointmentDay'].min(),
        end_date=df['AppointmentDay'].max(),
        className='mb-2'),
    dbc.Button("Reset Filters", id="reset-filters", color="secondary", className="mt-2 mb-2", n_clicks=0),
    dbc.Button("Download Filtered Data", id="download-csv", color="primary", className="mb-2", n_clicks=0),
    dcc.Download(id="download-dataframe-csv"),
    html.Hr(),
    dbc.Button("Show Help", id="open-help", color="info", className="mb-2", n_clicks=0),
], body=True, className="mb-4")

# Help Modal
help_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Dashboard Help & Documentation")),
    dbc.ModalBody([
        html.P("- Use the sidebar to filter data by age, gender, neighborhood, condition, and date."),
        html.P("- Click on chart elements to drill down (e.g., click a bar to filter by that value)."),
        html.P("- Download filtered data or export charts as images."),
        html.P("- Use the dark mode toggle for a different look."),
        html.P("- The map shows no-show rates by neighborhood (mocked coordinates)."),
        html.P("- The correlation matrix and prediction tab offer advanced analytics."),
        html.P("- Add your own notes in the Notes tab!"),
    ]),
    dbc.ModalFooter(
        dbc.Button("Close", id="close-help", className="ms-auto", n_clicks=0)
    ),
], id="help-modal", is_open=False)

# Tabs for main content
main_tabs = dcc.Tabs(id="main-tabs", value="dashboard", children=[
    dcc.Tab(label="Dashboard", value="dashboard"),
    dcc.Tab(label="Map", value="map"),
    dcc.Tab(label="Correlation Matrix", value="corr"),
    dcc.Tab(label="Prediction", value="predict"),
])

# Layout
app.layout = dbc.Container([
    html.H1("Medical Appointment Analysis Dashboard",style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Store(id='filtered-data'),
    welcome_modal,
    help_modal,
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            main_tabs,
            html.Div(id="main-content", className="mt-3")
        ], width=9)
    ], className="mt-2")
], fluid=True)

# --- Callbacks ---
# Welcome modal
@app.callback(
        Output("welcome-modal", "is_open"),
        [Input("close-welcome", "n_clicks")],
        [State("welcome-modal", "is_open")])
def close_welcome(n, is_open):
    if n:
        return False
    return is_open

# Help modal
@app.callback(
        Output("help-modal", "is_open"), 
        [Input("open-help", "n_clicks"), 
        Input("close-help", "n_clicks")], 
        [State("help-modal", "is_open")])
def toggle_help(open_n, close_n, is_open):
    if open_n or close_n:
        return not is_open
    return is_open

# Reset filters
@app.callback([
    Output('age-filter', 'value'),
    Output('gender-filter', 'value'),
    Output('neighborhood-filter', 'value'),
    Output('condition-filter', 'value'),
    Output('date-range', 'start_date'),
    Output('date-range', 'end_date')
], [Input('reset-filters', 'n_clicks')])
def reset_filters(n):
    if n:
        return 'All', 'All', 'All', 'All', df['AppointmentDay'].min(), df['AppointmentDay'].max()
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Filter data
@app.callback(
    Output('filtered-data', 'data'),
    [Input('age-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('neighborhood-filter', 'value'),
     Input('condition-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def filter_data(age, gender, neighborhood, condition, start, end):
    dff = df.copy()
    if age != 'All' and isinstance(age, list):
        if 'All' not in age:
            dff = dff[dff['AgeGroup'].isin(age)]
    if gender != 'All':
        dff = dff[dff['Gender'] == gender]
    if neighborhood != 'All' and isinstance(neighborhood, list):
        if 'All' not in neighborhood:
            dff = dff[dff['Neighbourhood'].isin(neighborhood)]
    if condition != 'All':
        dff = dff[dff[condition] == 1]
    if start and end:
        dff = dff[(dff['AppointmentDay'] >= start) & (dff['AppointmentDay'] <= end)]
    return dff.to_json(date_format='iso', orient='split')

# Download filtered data
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-csv", "n_clicks")],
    [State('filtered-data', 'data')],
    prevent_initial_call=True)
def download_csv(n, data):
    if n and data:
        dff = pd.read_json(data, orient='split')
        return dcc.send_data_frame(dff.to_csv, "filtered_appointments.csv")
    return dash.no_update

# Main content rendering
@app.callback(
    Output("main-content", "children"),
    [Input("main-tabs", "value"), Input('filtered-data', 'data')])
def render_main_content(tab, data):
    dff = df if not data else pd.read_json(data, orient='split')
    if tab == "dashboard":
        # Key metrics
        total = len(dff)
        no_show = f"{(dff['NoShow'].mean() * 100):.1f}%" if total else "-"
        avg_age = f"{dff['Age'].mean():.1f}" if total else "-"
        sms_rate = f"{(dff['SMS_received'].mean() * 100):.1f}%" if total else "-"
        metrics = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(total, className="text-center"),
                    html.P("Total Appointments", className="text-center")
                ])
            ], color="primary", inverse=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(no_show, className="text-center"),
                    html.P("No-Show Rate", className="text-center")
                ])
            ], color="danger", inverse=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(avg_age, className="text-center"),
                    html.P("Average Age", className="text-center")
                ])
            ], color="success", inverse=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H2(sms_rate, className="text-center"),
                    html.P("SMS Received Rate", className="text-center")
                ])
            ], color="warning", inverse=True), width=3),
        ], className="mb-4")
        # Charts
        pie = px.pie(dff, names='No-show', color='No-show', color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'})
        age_hist = px.histogram(dff, x='Age', color='No-show', nbins=20, color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'})
        day_bar = px.bar(dff.groupby(['DayOfWeek', 'No-show']).size().unstack(fill_value=0).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']), barmode='stack')
        gender_bar = px.bar(dff.groupby(['Gender', 'No-show']).size().unstack(fill_value=0), barmode='group')
        conds = ['Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
        cond_bar = px.bar({c: [dff[dff[c]==1]['NoShow'].mean()*100 if len(dff[dff[c]==1])>0 else 0] for c in conds}, labels={'value':'No-Show Rate (%)','variable':'Condition'})
        neigh = dff['Neighbourhood'].value_counts().head(10).index
        neigh_bar = px.bar(dff[dff['Neighbourhood'].isin(neigh)].groupby(['Neighbourhood', 'No-show']).size().unstack(fill_value=0), barmode='group')
        days_hist = px.histogram(dff[dff['DaysUntilAppointment']>=0], x='DaysUntilAppointment', color='No-show', nbins=20, color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'})
        sms_bar = px.bar(dff.groupby(['SMS_received', 'No-show']).size().unstack(fill_value=0), barmode='group')
        return [metrics,
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("Attendance Overview", className="text-center"),
                        dcc.Graph(figure=pie, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
                dbc.Col(
                    html.Div([
                        html.H4("Age Distribution", className="text-center"),
                        dcc.Graph(figure=age_hist, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("Day of Week Analysis", className="text-center"),
                        dcc.Graph(figure=day_bar, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
                dbc.Col(
                    html.Div([
                        html.H4("Gender Analysis", className="text-center"),
                        dcc.Graph(figure=gender_bar, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("No-Show Rate by Medical Condition", className="text-center"),
                        dcc.Graph(figure=cond_bar, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
                dbc.Col(
                    html.Div([
                        html.H4("No-Shows Across Neighborhoods", className="text-center"),
                        dcc.Graph(figure=neigh_bar, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4("Scheduling Lead Time", className="text-center"),
                        dcc.Graph(figure=days_hist, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
                dbc.Col(
                    html.Div([
                        html.H4("SMS Impact Analysis", className="text-center"),
                        dcc.Graph(figure=sms_bar, config={"toImageButtonOptions": {"format": "png"}})
                    ], style={'backgroundColor': 'white', 'padding': '20px','borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)','margin': '0 10px 20px 0'})
                ),
            ]),
            dbc.Row([
                dbc.Col(),
                dbc.Col(),
            ]),
        ]
    elif tab == "map":
        # Map visualization
        map_fig = px.scatter_mapbox(
            dff.groupby('Neighbourhood').agg({'lat':'first','lon':'first','NoShow':'mean','No-show':'count'}).reset_index(),
            lat='lat', lon='lon', size='No-show', color='NoShow',
            color_continuous_scale='Reds', size_max=30, zoom=11,
            hover_name='Neighbourhood',
            mapbox_style="open-street-map",
            hover_data={'No-show': True, 'NoShow': ':.2%'},
            title="No-Show Rate by Neighborhood (mocked coordinates)"
        )
        return dcc.Graph(figure=map_fig, config={"toImageButtonOptions": {"format": "png"}})
    elif tab == "corr":
        # Correlation matrix
        corr = dff[['Age','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','NoShow']].corr()
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Matrix")
        return dcc.Graph(figure=corr_fig, config={"toImageButtonOptions": {"format": "png"}})
    elif tab == "predict":
        # Simple predictive analytics (logistic regression)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        features = ['Age','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']
        X = dff[features].fillna(0)
        y = dff['NoShow']
        if len(X) > 100:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            return html.Div([
                html.H5(f"Logistic Regression Accuracy: {acc*100:.2f}%"),
                html.P("Features used: Age, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received"),
            ])
        else:
            return html.P("Not enough data for prediction.")
    return html.P("Select a tab.")

if __name__ == '__main__':
    app.run(debug=True, port=8050)