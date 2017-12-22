#					Computer Networks I
#===========================================================
#Plots generated usign plotly browser-based graphing library 
#===========================================================


import plotly
import plotly.plotly as py
import plotly.graph_objs as go

ECHO_PATH = "../session2/ECHO-2017_12_21_14_51_29.txt"

ARQ_PATH  = "../session2/ARQ-2017_12_21_14_55_51.txt"

def plotly_echo():

	#Get x and y data
	with open(ECHO_PATH) as f:
	    data = f.read()

	data = data.split('\n')

	my_x = [packet for packet in range (len(data)-6)]
	my_y = [int(row) for row in data[2:(len(data)-4)]]

	# Create a trace
	trace = go.Scatter(
		x = my_x,
		y = my_y
	) 

	data = [trace]

	layout = go.Layout(
	    title='Echo response time E6896 21/12/2017 14:51:29',
	    xaxis=dict(
	        title='Packets',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        title='Response time(millisec)',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)
	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='session2_echo')

def plotly_arq():

	#Get x and y data
	with open(ARQ_PATH) as f:
	    data = f.read()

	data = data.split('\n')

	my_x = [packet for packet in range (len(data)-6)]
	my_y = [int(row.split(' ')[1]) for row in data[2:(len(data)-4)]]


	# Create a trace
	trace = go.Scatter(
		x = my_x,
		y = my_y
	) 

	data = [trace]

	layout = go.Layout(
	    title='ACK response time Q9658 21/12/2017 14:55:51',
	    xaxis=dict(
	        title='Successful packets',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict
(	        title='Response time(millisec)',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)

	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='session2_ack')


def main():
	#API authentication
	plotly.tools.set_credentials_file(username='draptis', api_key='Ln4WaYXGVSrNmviQSQAv')

	plotly_echo()
	plotly_arq()

if __name__ == "__main__":
    main()