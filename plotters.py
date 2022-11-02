import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots #TODO add success heatmap/graph
from plotly.offline import plot #TODO check this out https://stackoverflow.com/a/58848335
from utils import * 


marker_symbols = ["circle", "x", "star","asterisk"]
curve_colors = px.colors.qualitative.Plotly


def plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1, enlarge_tests_num_by_factors, nmc, count_DD2, sample_method, method_DD, Tbaseline, typical_codes, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=count_DD2[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='DD(2), T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=count_PD1[:,idxT] - count_DD2[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='Unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))
    fig.add_trace(go.Scatter(x=vecK, y=vecK,
                            mode='lines+markers',
                            marker_line_color="white", marker_color="white",
                            name='K'))

    typical_label = 'no typical codes'
    if typical_codes:
        typical_label = 'typical codes'

    fig.update_layout(title=method_DD + ' DD || ' + sample_method + '<br>DD vs. K after CoMa and DD <br>\
                            N = ' + str(N) + ', T=T_{' + Tbaseline + '}*[' + str(enlarge_tests_num_by_factors) + '] <br>\
                            ' + typical_label + ', iterations=' + str(nmc),
                        xaxis_title='K',
                        yaxis_title='#DD',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')
    # fig.show()
    plot_and_save(fig, fig_name='DD_vs_K_and_T', results_dir_path=results_dir_path)

    
def plot_expected_DD(vecK, expected_DD, real_DD, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_DD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected DD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_DD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated DD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected DD vs. simulated DD',
                        xaxis_title='K',
                        yaxis_title='#DD(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_DD', results_dir_path=results_dir_path)

def plot_expected_PD(vecK, expected_PD, real_PD, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_PD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected PD, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))

        fig.add_trace(go.Scatter(x=vecK, y=real_PD[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated PD, T=' + str(enlarge_tests_num_by_factors[idxT])+ 'T = ' + str(T)),)

    fig.update_layout(title='Expected PD vs. simulated PD',
                        xaxis_title='K',
                        yaxis_title='#PD(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_PD', results_dir_path=results_dir_path)
    
def plot_expected_unknown(vecK, expected_unknown, real_unknown, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected unknown vs. simulated unknown',
                        xaxis_title='K',
                        yaxis_title='#unknown(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_unknown', results_dir_path=results_dir_path)

def plot_expected_not_detected(vecK, expected_not_detected, real_not_detected, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_not_detected[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected not detected, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_not_detected[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated not detected, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected not detected vs. simulated not detected',
                        xaxis_title='K',
                        yaxis_title='#not_detected',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_not_detedcted', results_dir_path=results_dir_path)

def plot_expected_unknown_avg(vecK, expected_unknown, real_unknown, vecT, enlarge_tests_num_by_factors, results_dir_path=None):
    fig = go.Figure()
    for idxT, T in enumerate(vecT):
        fig.add_trace(go.Scatter(x=vecK, y=expected_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                hovertemplate='%{y:.3f}',
                                name='Expected unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)))
                                # [curveStyles(1), markerStyles(1), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT)))
        fig.add_trace(go.Scatter(x=vecK, y=real_unknown[:,idxT], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxT], marker_color=curve_colors[-idxT],
                                line_dash='dash',
                                hovertemplate='%{y:.3f}',
                                name='simulated unknown, T=' + str(enlarge_tests_num_by_factors[idxT]) + 'T = ' + str(T)),)
                                # [curveStyles(1), markerStyles(2), curve_colors(idxT)], 'MarkerFaceColor',curve_colors(idxT))

    fig.update_layout(title='Expected unknown vs. simulated unknown (averaged)',
                        xaxis_title='K',
                        yaxis_title='#unknown(2)',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='expected_unknown_average', results_dir_path=results_dir_path)

def plot_Psuccess_vs_T(vecTs, count_success_DD, count_success_Tot, vecK, results_dir_path):
    fig = go.Figure()
    for idxK,K in enumerate(vecK):
        vecT = vecTs[idxK]
        fig.add_trace(go.Scatter(x=vecT, y=count_success_DD[idxK,:], 
                                mode='lines+markers',
                                marker_symbol=marker_symbols[0],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxK], marker_color=curve_colors[-idxK],
                                hovertemplate='%{y:.3f}',
                                name='Psuccess using DD, K=' + str(K)))
        
        fig.add_trace(go.Scatter(x=vecT, y=count_success_Tot[idxK,:], 
                                mode='lines+markers',
                                line_dash='dash',
                                marker_symbol=marker_symbols[1],
                                marker_size=10,
                                marker_line_color=curve_colors[-idxK], marker_color=curve_colors[-idxK],
                                hovertemplate='%{y:.3f}',
                                name='Psuccess Tot, K=' + str(K)))

    fig.update_layout(title='Probability of success vs. T  <br>\
                            (success in terms of exact analysis)',
                        xaxis_title='T',
                        yaxis_title='#Ps [%]',
                        hovermode="x",
                        hoverlabel = dict(namelength = -1, font_size=16),
                        template='plotly_dark')

    plot_and_save(fig, fig_name='Ps_vs_T', results_dir_path=results_dir_path)

def plot_and_save(fig, fig_name, results_dir_path=None):
    if results_dir_path is not None:
        fig.write_html(os.path.join(results_dir_path, fig_name+'.html'), auto_open=True)
    else:
        fig.show()


if __name__ == '__main__':
    db_path=r'/Users/ayelet/Library/CloudStorage/OneDrive-Technion/Alejandro/count_possibly_defected_results/shelve_raw/countPDandDD_N100_nmc1000_methodDD_Sum_typical_Tbaseline_ML_07082022_092256.mat'
    var_dict = load_workspace(db_path)
    for key in var_dict.keys():
        globals()[key] = var_dict[key]
    plot_DD_vs_K_and_T(N, vecT, vecK, count_PD1, enlarge_tests_num_by_factors, nmc, count_DD2, sample_method, method_DD, Tbaseline, typical_codes)
    pass