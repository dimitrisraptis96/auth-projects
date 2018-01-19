import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def node_1():
    n2  = (35.167358 + 38.410144 + 33.344922 + 42.399180)/4
    n4  = (23.285071 + 20.887386 + 21.228797 + 23.135862)/4
    n8  = (11.740781 + 12.181761 + 11.503121 + 12.147270)/4
    n16 = (8.920446 + 10.119909 + 9.716439 + 9.470424)/4
    n32 = (11.278722 + 11.250713 + 10.516995 + 11.275163)/4

    n2_  = (55.954626 + 33.629328 + 54.578408 + 40.049635)/4
    n4_ = (34.574359 + 29.862025 + 29.817031 + 30.899485)/4
    n8_  = (16.738622 + 16.037621 + 11.255267 + 13.146556)/4
    n16_ = (11.694875 + 10.683421 + 9.800897  + 10.120483)/4
    n32_ = (11.897087 + 10.503296 + 10.160746 + 10.003062)/4

    serial = (218.415197 + 238.176272 + 181.325651 + 199.650865)/4

    node =1
    cores = [2, 4, 8, 16, 32]
    block_y = [n2,n4,n8,n16,n32]
    no_block_y=[n2_,n4_,n8_,n16_,n32_]
    serial=[serial, serial, serial, serial,serial]

    trace1 = go.Scatter(
        x = cores,
        y = block_y,
        mode = 'lines+markers',
        name = 'Blocking MPI'
    )

    trace2 = go.Scatter(
        x = cores,
        y = no_block_y,
        mode = 'lines+markers',
        name = 'Non-blocking MPI'
    )

    trace3 = go.Scatter(
        x = cores,
        y = serial,
        mode = 'lines+markers',
        name = 'Serial'
    )

    data = [trace1,trace2, trace3]

    layout = go.Layout(
        title='Hellasgrid nodes used: n = 1',
        xaxis=dict(
            title='Number of processors',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Time measurement (sec)',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename='node-1')
    
def node_2():
    n2  = (37.610656 + 27.546454 + 29.587619 + 38.671292)/4
    n4  = (18.798068  +  18.394992   +15.501010 +14.433781)/4
    n8  = (9.077509 +9.235159  +  8.750739  +  8.534303)/4
    n16 = (8.098234 +8.746313  +  9.236239  +  8.924647)/4
    n32 = (8.119225 +9.089740  +  8.711368  +  8.240480)/4

    n1_  = (44.912027 +  46.525639  + 46.332636  + 45.798293)/4
    n2_  = (36.594086  + 32.415354 +  34.209401 +  32.336408)/4
    n4_  = (12.099364  + 11.309187  + 7.144368   + 10.306163 )/4
    n8_ = (7.414768   + 8.268169   + 7.951037   + 7.886487)/4
    n16_ = (8.367700   + 7.729454   + 8.487257   + 7.819741)/4

    serial = (218.415197 + 238.176272 + 181.325651 + 199.650865)/4

    processes = [2, 4, 8, 16, 32]
    block_y = [n2,n4,n8,n16,n32]
    no_block_y=[n1_,n2_,n4_,n8_,n16_]
    serial=[serial, serial, serial, serial,serial]

    trace1 = go.Scatter(
        x = processes,
        y = block_y,
        mode = 'lines+markers',
        name = 'Blocking MPI'
    )

    trace2 = go.Scatter(
        x = processes,
        y = no_block_y,
        mode = 'lines+markers',
        name = 'Non-blocking MPI'
    )

    trace3 = go.Scatter(
        x = processes,
        y = serial,
        mode = 'lines+markers',
        name = 'Serial'
    )

    data = [trace1,trace2, trace3]

    layout = go.Layout(
        title='Hellasgrid nodes used: n = 2',
        xaxis=dict(
            title='Number of processes',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Time measurement (sec)',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename='node-2')

def main():
    plotly.tools.set_credentials_file(username='draptis', api_key='VMOhSrokZOzT6DiVXnOU')
    node_1()
    node_2()

if __name__ == "__main__":
    main()