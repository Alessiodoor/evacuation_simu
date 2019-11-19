import pprint
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt

# Plotly
import plotly.plotly as py
import plotly.tools as tls
# Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt

# Buff
import pexpect
import io
import base64

#file 
import os

import random

def Rand(start, end, num): 
    res = [] 
  
    for j in range(num): 
        res.append(str(random.randint(start, end)))
  
    return res 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def_img = 'event_cat_domain_wnum.png'
encoded_image = base64.b64encode(open(def_img, 'rb').read())

#inizializzazione variabili
Tf = 50

#valori bottoni
prev_all = 0
prev_no = 0
prev_10 = 0
prev_20 = 0
prev_30 = 0

#creo la lista di checkbox per l'uni
uni_pos_options = []
for i in range(1, 71):
    uni_pos_options.append({'label': str(i), 'value': str(i)})

uni_pos_values = [str(i) for i in range(0, 71)]

app.title = 'Simulatore'
app.layout = html.Div(children=[ 
    html.H2(
        children='Simulatore di evacuazioni',
        style = {'text-align': 'center'}),

    #set
    html.Div([
        html.Div(
            id = 'set-div',
            children = [
            html.Div([
                html.H3(
                    'Modello',
                    style = {'width': '100%', 'display': 'inline-block', 'margin-bottom': '20px', 'float': 'left'}
                ),
                dcc.RadioItems(
                    id='model-dropdown',
                    options=[
                        {'label': 'Automa cellulare', 'value': 'automa'},
                        {'label': 'Granulare', 'value': 'granular'},
                        {'label': 'Forze sociali', 'value': 'social'}
                    ],
                    value='automa',
                    style = {'width': '100%', 'float': 'left'}
                ),
                html.H3(
                    'Ambiente',
                    style = {'width': '100%', 'display': 'inline-block', 'margin-bottom': '20px', 'float': 'left'}
                ),
                dcc.RadioItems(
                    id = 'ev_radio',
                    options=[
                        {'label': 'Scuola Elementare Catanzaro', 'value': 'cat'},
                        {'label': 'Inner Mongolia University', 'value': 'uni'}
                    ],
                    value='cat',
                    style ={'width': '60%', 'float': 'left', 'margin-bottom': '20px'}
                ),
                html.Img(
                    id='ev_img',
                    src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
                    style = {'width': 'auto', 'height': '200px', 'float': 'left', 'margin-right': '10px', 'margin-left': '10px'}
                ),
                html.Div(
                    id = 'uni_pos_chbox-div',
                    children = [
                        dcc.Checklist(
                            id = 'uni_pos_chbox',
                            options = uni_pos_options,
                            values = uni_pos_values,
                            labelStyle = {'display': 'inline-block'}
                        ),
                        html.Div(
                            children = [
                                html.Button('Tutti', id='all', style = {'margin-right': '10px', 'margin-bottom': '10px'}),
                                html.Button('Nessuno', id='no', style = {'margin-right': '10px', 'margin-bottom': '10px'}),
                                html.Button('10 alunni casuali', id = '10c', style = {'margin-right': '10px', 'margin-bottom': '10px'}),
                                html.Button('20 alunni casuali', id = '20c', style = {'margin-right': '10px'}),
                                html.Button('30 alunni casuali', id = '30c'),
                            ],
                            style = {'margin-top': '20px'}
                        ),
                    ],
                    style = {'width': '450px', 'margin-top': '500px', 'margin-left': '10px'}
                ),
                html.H3(
                    id = 'density-title-div',
                    children = 'Numero di alunni',
                    style = {'width': '100%', 'display': 'inline-block', 'float': 'left'}
                ),
                html.Div(
                    id = 'density-div',
                    children = [
                        dcc.RadioItems(
                        id='density-radio',
                        options=[
                            {'label': 'Uno per banco (9)', 'value': '10'},
                            {'label': 'Due per banco (18)', 'value': '19'},
                            {'label': 'Tre per banco (27)', 'value': '28'}
                        ],
                        value = '19',
                        style = {'width': '100%', 'float': 'left'}
                    )]
                )],
                style = {'width': '100%', 'display': 'inline-block', 'margin-bottom': '20px'}
            )
        ], style = {'background-color': '#f2f2f2', 'border-radius': '4px', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '50px', 'width': '40%', 'display': 'inline-block', 'float': 'left', 'margin-right': '2%'}
        ),
        #fine set

        #par
            html.Div([
                html.H3('Regola i parametri del modello'),
                html.Div(
                    id = 'vmax_div',
                    children = [
                    html.P('Velocità desiderata',
                        style = {'margin-top': '10px'}),
                    dcc.Slider(
                        id='vmax_slider',
                        min=0.5,
                        max=1.5,
                        step=0.25,
                        value=1,
                        marks={str(round(i, 3)): str(round(i, 3)) for i in np.arange(0.5, 1.75, 0.25)}
                    )
                ]),
                html.Div(
                    id = 'dmin_div',
                    children = [
                    html.P('Dmin',
                        style = {'margin-top': '80px'}),
                    dcc.Slider(
                        id='dmin_slider',
                        min=0,
                        max=0.5,
                        step=0.05,
                        value=0,
                        marks={str(round(i, 3)): str(round(i, 3)) for i in np.arange(0, 0.51, 0.05)}
                    )
                ]),
                html.Div(
                    id = 'F_div',
                    children = [
                        html.P('F', style = {'margin-top': '80px'}),
                        dcc.Slider(
                            id='F_slider',
                            min=500,
                            max=4000,
                            step=500,
                            value=2000,
                            marks={str(i): str(i) for i in np.arange(500, 4500, 500)}
                        )
                ]),
                html.Div(
                    id = 'FWall_div',
                    children = [
                        html.P('FWall', style = {'margin-top': '80px'}),
                        dcc.Slider(
                            id='FWall_slider',
                            min=500,
                            max=4000,
                            step=500,
                            value=1000,
                            marks={str(i): str(i) for i in np.arange(500, 4500, 500)}
                        )
                ]),
                html.Div(
                    id = 'Lam_div',
                    children = [
                        html.P('Lambda', style = {'margin-top': '80px'}),
                        dcc.Slider(
                            id='Lam_slider',
                            min=0.125,
                            max=1,
                            step=0.125,
                            value=0.125,
                            marks={str(i): str(i) for i in np.arange(0.125, 1.125, 0.125)}
                        )
                ]),

                html.Div(
                    id = 'tau_div',
                    children = [
                        html.P('Tau', style = {'margin-top': '80px'}),
                        dcc.Slider(
                            id='tau_slider',
                            min=0.05,
                            max=0.15,
                            step=0.01,
                            value=0.05,
                            marks={str(round(i, 3)): str(round(i, 3)) for i in np.arange(0.05, 0.16, 0.01)}
                        )
                ]),

                html.Div(
                    id = 'movd_div',
                    children = [
                        html.P('Movimento', style = {'margin-top': '80px'}),
                        dcc.Dropdown(
                            id='movd-dropdown',
                            options=[
                                {'label': 'Vicinato di von Neumann', 'value': 'False'},
                                {'label': 'Vicinato di Moore', 'value': 'True'},
                            ],
                            value='False',
                            style = {'width': '100%'}
                        )
                ])

                ],
                style = {'background-color': '#f2f2f2', 'border-radius': '4px', 'box-shadow': '2px 2px 2px lightgrey', 'width': '40%', 'padding': '50px', 'display': 'inline-block', 'float': 'left', 'margin-right': '10px'}
            ),
        #fine par
    ]),

    html.Div(
        id = 'vis-div',
        children = [
        html.H3('Regolare la visualizzazione',
            style = {'margin-top': '20px'}
        ),
        html.P('Intervallo di visualizzazione'),
        dcc.Slider(
            id='drawper_slider',
            min=10,
            max=50,
            step=10,
            value=30,
            marks={str(i): str(i) for i in np.arange(10, 60, 10)}
        )],
        style = {'background-color': '#f2f2f2', 'border-radius': '4px', 'padding': '50px', 'box-shadow': '2px 2px 2px lightgrey', 'width': '40%', 'display': 'block', 'float': 'left', 'margin-top': '20px'}
    ),

    html.Div(
        children = [
            html.Button(
                'Start', 
                id='path_submit',
                n_clicks = 0,
                style = {'background-color': 'rgb(171, 225, 251)', 'width': '150px', 'height': '50px', 'margin-left': '45%'}
            ),
            html.P(id = 'msg', style = {'text-align': 'center', 'color': 'red'})
        ],
        style = {'width': '100%', 'height': '100px', 'padding': '50px', 'display': 'inline-block', 'float': 'left', 'margin-top': '20px'}
    ),

    html.Div(
        children = [
            html.H3(
                'Istanti simulazione'
            ),
            html.Div(
                html.Div(
                    id = 'img-container'
                ),
                style = {'width': '100%', 'display': 'block', 'margin-bottom': '20px'}
            )
        ],
        style = {'width': '42%', 'float': 'left', 'background-color': '#f2f2f2', 'padding-left': '50px', 'border-radius': '4px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-right': '2%'}
    ),

    html.Div(
        children = [
            html.H2(
                'Risultati'
            ),
            html.Div(
                html.Div(
                    id = 'result-container'
                ),
                style = {'width': '100%', 'display': 'block', 'margin-bottom': '20px'}
            )
        ],
        style = {'width': '40%', 'float': 'left', 'background-color': '#f2f2f2', 'padding-left': '50px', 'border-radius': '4px', 'box-shadow': '2px 2px 2px lightgrey', 'margin-bottom': '300px'}
    )],

    style = {'padding': '70px'}
)

@app.callback(
    dash.dependencies.Output('uni_pos_chbox', 'values'),
    [dash.dependencies.Input('all', 'n_clicks'),
    dash.dependencies.Input('no', 'n_clicks'),
    dash.dependencies.Input('10c', 'n_clicks'),
    dash.dependencies.Input('20c', 'n_clicks'),
    dash.dependencies.Input('30c', 'n_clicks')])
def clicked_all(all_c, no_c, c_10, c_20, c_30):
    global prev_all 
    global prev_no
    global prev_10
    global prev_20
    global prev_30
    values = []
    if all_c is not None:
        if all_c > prev_all:#cliccato 'tutti'
            values = [str(i) for i in range(1, 71)]
            prev_all = all_c
    if no_c is not None:   
        if no_c > prev_no:#cliccato 'nessuno'
            prev_no = no_c

    if c_10 is not None:
        if c_10 > prev_10:
            prev_10 = c_10
            values = Rand(1, 70, 10)

    if c_20 is not None:
        if c_20 > prev_20:
            prev_20 = c_20
            values = Rand(1, 70, 20)

    if c_30 is not None:
        if c_30 > prev_30:
            prev_30 = c_30
            values = Rand(1, 70, 30)

    return values


@app.callback([
    dash.dependencies.Output('vmax_div', 'style'),
    dash.dependencies.Output('F_div', 'style'),
    dash.dependencies.Output('FWall_div', 'style'),
    dash.dependencies.Output('Lam_div', 'style'),
    dash.dependencies.Output('tau_div', 'style'),
    dash.dependencies.Output('movd_div', 'style'),
    dash.dependencies.Output('dmin_div', 'style')],
    [dash.dependencies.Input('model-dropdown', 'value')])
def change_par(model):
    style = []
    if model == 'granular':
        style = []
        style.append({'display': 'block', 'width': '100%'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'block', 'width': '100%'})

    if model == 'social':
        style = []
        style.append({'display': 'block', 'width': '100%'})
        style.append({'display': 'block', 'width': '45%', 'float': 'left'})
        style.append({'display': 'block', 'width': '45%', 'float': 'right'})
        style.append({'display': 'block', 'width': '45%', 'float': 'left'})
        style.append({'display': 'block', 'width': '45%', 'float': 'right'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})

    if model == 'automa':
        style = []
        style.append({'display': 'block', 'width': '100%'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'none'})
        style.append({'display': 'block'})
        style.append({'display': 'none'})

    return style

@app.callback(
    [dash.dependencies.Output('ev_img', 'src'),
    dash.dependencies.Output('density-div', 'style'),
    dash.dependencies.Output('density-title-div', 'style'),
    dash.dependencies.Output('uni_pos_chbox-div', 'style')],
    [dash.dependencies.Input('ev_radio', 'value')])
def update_ev(name):
    def_img = 'event_' + name + '_domain_wnum.png'
    encoded_image = base64.b64encode(open(def_img, 'rb').read())
    style_dens = {}
    style_ck = {}
    if name =='cat':
        style_dens = {'display': 'block'}
        style_ck = {'width': '450px', 'margin-top': '500px', 'margin-left': '10px', 'display': 'none'}
    else:
        style_dens = {'display': 'none'} 
        style_ck = {'width': '450px', 'margin-top': '500px', 'margin-left': '10px', 'display': 'block'}


    return ['data:image/png;base64,{}'.format(encoded_image.decode()), style_dens, style_dens, style_ck]

@app.callback(
    [dash.dependencies.Output("img-container", "children"),
    dash.dependencies.Output('result-container', "children"),
    dash.dependencies.Output('msg', 'children')],
    [dash.dependencies.Input('path_submit', 'n_clicks')],
    [dash.dependencies.State('vmax_slider', 'value'),
    dash.dependencies.State('ev_radio', 'value'),
    dash.dependencies.State('drawper_slider', 'value'),
    dash.dependencies.State('model-dropdown', 'value'),
    dash.dependencies.State('F_slider', 'value'),
    dash.dependencies.State('FWall_slider', 'value'),
    dash.dependencies.State('Lam_slider', 'value'),
    dash.dependencies.State('tau_slider', 'value'),
    dash.dependencies.State('movd-dropdown', 'value'),
    dash.dependencies.State('density-radio', 'value'),
    dash.dependencies.State('uni_pos_chbox', 'values'),
    dash.dependencies.State('dmin_slider', 'value')])
def update_output(disabled, v_max, fileName, drawper, model, F, FWall, lam, tau, movd, density, ck_values, dmin):
    if len(ck_values) == 0 and model == 'uni':
        return [' ', ' ', 'Seleziona almeno un posto']
    else:
        if disabled:
            aus = ""
            for n in ck_values:
                aus += n + " "

            p_index = aus

            p_index = "'" + p_index + "'"

            parameters = {}
            parameters['granular'] = [('vmax', v_max), ('drawper', drawper), ('dmin', dmin)]
            parameters['social'] = [('vmax', v_max), ('F', F), ('FWall', FWall), ('lambda', lam), ('tau', tau), ('drawper', drawper)]
            parameters['automa'] = [('vmax', v_max), ('movd', movd), ('drawper', drawper)]
            #rimuovo immaigni vecchie
            mydir = 'results/'
            filelist = [ f for f in os.listdir(mydir) if f.endswith(".png") ]
            for f in filelist:
                os.remove(os.path.join(mydir, f))

            if model == 'granular':
                program = model + '/exe.py'
            else:
                program = model + '/demo_' + str(model) + '_' + fileName + ".py"

            options = ''

            if fileName == 'cat':
                parameters[model].append(('Np', density))

            if fileName == 'uni':
                parameters[model].append(('p_index', p_index))

            for k in parameters[model]:
                options += ' --' + str(k[0]) + ' ' + str(k[1])

            command = "python3 " + program + " --json " + model + "/input_" + fileName + ".json " + options

            print(command)
            #pexpect.run(command)
            child = pexpect.spawn(command)
            child.expect(pexpect.EOF, timeout=None)

            stop = False
            time = drawper / 10
            images = int(Tf / time)
            imgs = []
            for i in range(0, images):
                counter = str(i * drawper).zfill(6)
                image_path = "results/fig_" + counter + ".png"
                #print(image_path)
                if not stop:
                    try:
                        encoded_image = base64.b64encode(open(image_path, 'rb').read())
                        src = 'data:image/png;base64,{}'.format(encoded_image.decode())
                    except:
                        stop = True
                if stop:
                    #print(image_path)
                    #def_img = 'event_' + fileName + '_domain.png'
                    #encoded_image = base64.b64encode(open(def_img, 'rb').read())
                    src = ''
                imgs.append(html.Img(
                    id='ev_img' + str(i),
                    src = src,
                    style = {'width': '30%', 'height': 'auto', 'margin-right': '2%'}
                ))

            #appendo i percorsi
            imgs.append(html.H3('Percorsi'))

            image_path = "results/fig_paths.png"
            try:
                encoded_image = base64.b64encode(open(image_path, 'rb').read())
                src = 'data:image/png;base64,{}'.format(encoded_image.decode())
            except:
                print('Image not found')

            imgs.append(html.Img(
                id='ev_img' + str(i),
                src = src,
                style = {'width': '50%', 'height': 'auto', 'margin': '0 auto'}
            ))  

            results = []     
            f = open('plot_data.txt', 'r')
            content = f.readlines()

            for row in content:
                if 'mean ev' in row:
                    stamp = 'Tempo di evacuazione: ' + row.split(" ")[-1][0: 5] + ' secondi'
                    results.insert(0, html.Div(
                        children = [
                            html.H3(children = stamp)
                        ])
                    )
                if 'flows per ogni porta' in row:
                    f = eval(row.split('@')[-1])
                    plots = []

                    for door in f:
                        flows = {}
                        flows['x'] = [i for i in range(0, len(f[door]))]
                        flows['y'] = f[door]
                        flows['type'] = 'linear'
                        flows['name'] = 'Porta ' + str(door)
                        plots.append(flows)

                    results.append(dcc.Graph(
                                id='flow-graph',
                                figure={
                                    'data': plots,
                                    'layout': {
                                        'title': 'Flusso di evacuazione'
                                    }
                                }
                            )
                    )
                if 'time_single_unit' in row:
                    col = ['Posizione', 'Tempo arrivo']
                    t = eval(row.split('@')[-1])
                    data = []
                    for k in t:
                        data.append({'Posizione': str(k), 'Tempo arrivo': str(round(t[k][0], 3))})

                    results.append(html.H3('Tempo di arrivo per ogni posizione'))
                    results.append(dt.DataTable(
                        id = 'table',
                        columns = [{"name": i, "id": i} for i in col],
                        data = data,
                        style_header = {
                            'backgroundColor': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell = {'textAlign': 'left'},
                        style_as_list_view = True,
                        style_data_conditional = [
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                    ))

            return [imgs, results, ' ']
        else: 
            return [' ', ' ', ' ']

if __name__ == '__main__':
    app.run_server(debug=True)