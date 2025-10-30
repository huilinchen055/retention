from pydomo import Domo
import pandas as pd
from io import StringIO

client_id = "d9e4748c-c9ea-4ae6-99b4-95ffa839de02"
secret = "f505872f34e707fe4b2fec2e81da80d7f2a47ef72b60707eb3d7fbf93445b4be"

api = Domo(client_id, secret, api_host='api.domo.com') 

dataset_id = '66ac46a6-c72b-4fca-918a-14342ed4183b'
#dataset_id = '7f8007bd-9d50-4064-9fec-0e8739e23ecb'

# Export dataset data
csv_data = api.datasets.data_export(dataset_id, include_csv_header=True)

# Convert to pandas DataFrame
df = pd.read_csv(StringIO(csv_data))

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

app = Dash(__name__, routes_pathname_prefix='/retention/')

df = pd.DataFrame({
    'date': pd.date_range(start='2024-01-01', periods=100),
    'sales': range(100, 200)
})

app.layout = html.Div([
    html.H1('Date Range Dashboard'),
    
    dcc.DatePickerRange(
        id='date-picker',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD'
    ),
    
    dcc.Graph(id='filtered-graph')
])

@callback(
    Output('filtered-graph', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    fig = px.line(filtered_df, x='date', y='sales', title='Filtered Sales Data')
    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)