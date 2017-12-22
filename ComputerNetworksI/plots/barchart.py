import plotly
import plotly.plotly as py
import plotly.graph_objs as go

ARQ_PATH  = "../session2/ARQ-2017_12_21_14_55_51.txt"


def plotly_arq():

	#Get x and y data
	with open(ARQ_PATH) as f:
	    data = f.read()

	data = data.split('\n')

	my_x = [packet for packet in range (len(data)-6)]
	my_y = [int(row.split(' ')[2]) for row in data[2:(len(data)-4)]]

	zeroes = ones = twoes =threes =more = 0

	#Calculate the number of resends
	for y in my_y:
		if y == 0:
			zeroes+=1
		elif y == 1:
			ones+=1
		elif y == 2:
			twoes+=1
		elif y == 3:
			threes+=1
		else:
			more+=1

	# Create a trace
	trace = go.Bar(
		x = ['0', '1', '2', '3', '4'],
		y = [zeroes, ones, twoes, threes, more]
	) 

	data = [trace]

	layout = go.Layout(
	    title='ACK bar chart Q9658 21/12/2017 14:55:51',
	    xaxis=dict(
	        title='resends',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	    	title='frequency (Total packets 318)',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)

	fig = go.Figure(data=data, layout=layout)
	plot_url = py.plot(fig, filename='session2_resends')


def main():
	#API authentication
	plotly.tools.set_credentials_file(username='draptis', api_key='Ln4WaYXGVSrNmviQSQAv')

	plotly_arq()

if __name__ == "__main__":
    main()