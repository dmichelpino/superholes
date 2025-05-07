#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:59:13 2022

@author: D. Michel Pino
@email: dmichel.pino@csic.es
"""


import matplotlib.pyplot as p
import numpy as np

class var:
    def __init__(self, name, array, xlabel, norm=1, xarr=False, form='{:.2f}'):
        self.name = name
        self.array = array
        self.xlabel = xlabel
        self.norm = norm
        if type(xarr) == bool:
            self.xarr = array
        else:
            self.xarr = xarr
        self.form = form

def plot(var, params, pargs, padd=None, psave=None, label=None):
    pargs = pargs.copy()
    fig, ax = p.subplots(figsize=(6,4))
    
    match 'scatter' in pargs:
        case False:
            match len(var):
                case 2:
                    match len(var[1].array.shape):
                        case 1:
                             if type(pargs['color']) == list:
                                 pargs['color'] = pargs['color'][0]
                             ax.plot(var[0].array/var[0].norm, var[1].array/var[1].norm, color=pargs['color'], linewidth=pargs['linewidth'], dashes=pargs['dashes'])
                        case _:
                            if type(pargs['color']) != list:
                                pargs['color'] = [pargs['color']]*var[1].array.shape[0]
                            if type(pargs['linewidth']) != list:
                                pargs['linewidth'] = [pargs['linewidth']]*var[1].array.shape[0]
                            if len(pargs['dashes']) == 0:
                                pargs['dashes'] = [pargs['dashes']]*var[1].array.shape[0]
                            if 'alpha' not in pargs:
                                pargs['alpha'] = [None]*var[1].array.shape[0]
                            match len(var[0].array.shape):
                                case 1:
                                    for i in range(var[1].array.shape[0]):
                                        ax.plot(var[0].array/var[0].norm, var[1].array[i]/var[1].norm, color=pargs['color'][i], linewidth=pargs['linewidth'][i], dashes=pargs['dashes'][i], alpha=pargs['alpha'][i])
                                case _:
                                    for i in range(var[1].array.shape[0]):
                                        ax.plot(var[0].array[i]/var[0].norm, var[1].array[i]/var[1].norm, color=pargs['color'][i], linewidth=pargs['linewidth'][i], dashes=pargs['dashes'][i], alpha=pargs['alpha'][i])
                    im = False
                case 3:
                    if 'vmin' not in pargs:
                        pargs['vmin'] = None
                        pargs['vmax'] = None
                        
                    im = ax.pcolormesh(var[0].array/var[0].norm, var[1].array/var[1].norm, var[2].array/var[2].norm, cmap=pargs['cmap'], vmin=pargs['vmin'], vmax=pargs['vmax'])
        case True:
            if 'vmin' not in pargs:
                pargs['vmin'] = None
                pargs['vmax'] = None
            match len(var[1].array.shape):
                case 1:
                    im = ax.scatter(var[0].array/var[0].norm, var[1].array/var[1].norm, c=var[2].array/var[2].norm, cmap=pargs['cmap'], s=pargs['linewidth'], vmin=pargs['vmin'], vmax=pargs['vmax'])
                case _:
                    for i in range(var[1].array.shape[0]):
                        im = ax.scatter(var[0].array/var[0].norm, var[1].array[i]/var[1].norm, c=var[2].array[i]/var[2].norm, cmap=pargs['cmap'], s=pargs['linewidth'], vmin=pargs['vmin'], vmax=pargs['vmax'])
            
    if padd != None:
        plot_additional(ax, var, padd, pargs, label)
    else:
        padd = dict()
        plot_additional(ax, var, padd, pargs, label)
       
    plot_pretty(fig, ax, var, params, pargs, im, psave)
    
    return 0
    
def plot_save(fig, var, params, psave):
    save = psave['main_path']
    N = len(var)
    ptype = ['lp', 'cp']
    vname = ''
    for i in range(N-1):
        vname += var[i].name + '_'
    
    prms = ''
    if 'var_disc' in psave:
        prms += '_'+psave['var_disc']['name']
        for i in range(len(psave['var_disc']['array'])):
            prms += '-'+str(round(psave['var_disc']['array'][i],2))
        for name,value in params.items():
            if name != var[0].name and name != var[1].name and type(value) != str:
                prms += '_' + name + '-' + str(round(value,2))
    else:
        for name,value in params.items():
            if name != var[0].name and name != var[1].name and type(value) != str:
                if type(value) == list:
                    prms += '_' + name
                    for i in range(len(value)):
                        prms += '-' + str(round(value[i],2))
                else:
                    prms += '_' + name + '-' + str(round(value,2))
    
    save = psave['main_path'] + '{ptype}_{qname}-{system}__{vname}_{prms}.{fformat}'.format(ptype=ptype[N-2], qname=var[N-1].name, system=params['system'], vname=vname, prms=prms, fformat=psave['file_format'])
    
    fig.savefig(save, format=psave['file_format'], bbox_inches='tight', dpi=600, transparent=True)
    return 0

def plot_pretty(fig, ax, var, params, pargs, im=False, psave=None):
    if 'aspect' in pargs:
        ax.set_box_aspect(pargs['aspect'])
    else:
        ax.set_box_aspect(3/4.5)
    # ax.set_box_aspect(2/4)
    if 'ylim' not in pargs:
        pargs['ylim'] = [np.min(var[1].array)/var[1].norm, np.max(var[1].array)/var[1].norm]
    if 'xlim' not in pargs:
        pargs['xlim'] = [np.min(var[0].array)/var[0].norm, np.max(var[0].array)/var[0].norm]
    p.setp(ax, xlim = pargs['xlim'], ylim = pargs['ylim'], xscale=pargs['xscale'], yscale=pargs['yscale'])
    if 'xticks' in pargs:
        if 'xticklabels' not in pargs:
            pargs['xticklabels'] = []
            for i in range(len(pargs['xticks'])):
                pargs['xticklabels'].append(str(pargs['xticks'][i]))
        p.setp(ax, xticks=pargs['xticks'], xticklabels=pargs['xticklabels'])
    if 'yticks' in pargs:
        if 'yticklabels' not in pargs:
            pargs['yticklabels'] = []
            for i in range(len(pargs['yticks'])):
                pargs['yticklabels'].append(str(pargs['yticks'][i]))
        p.setp(ax, yticks=pargs['yticks'], yticklabels=pargs['yticklabels'])
    if 'yaxisright' in pargs:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    if 'xaxistop' in pargs:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    if 'xticksalignment' in pargs:
        p.setp( ax.xaxis.get_majorticklabels(), ha=pargs['xticksalignment'] )
    if 'yticksalignment' in pargs:
        p.setp( ax.yaxis.get_majorticklabels(), va=pargs['yticksalignment'] )
    
    if 'hidexaxis' in pargs:
        ax.tick_params(axis='x', labelcolor=[1,0,0,0])
        ax.xaxis.label.set_color([1,0,0,0])
    if 'hideyaxis' in pargs:
        ax.tick_params(axis='y', labelcolor=[1,0,0,0])
        ax.yaxis.label.set_color([1,0,0,0])
    
    if type(ax)==np.ndarray:
        for axs in ax:
            axs.tick_params(labelsize=16)
            axs.set_xlabel(var[0].xlabel, fontsize=16)
            axs.set_ylabel(var[1].xlabel, fontsize=16)
    else:
        ax.tick_params(labelsize=16)
        ax.set_xlabel(var[0].xlabel, fontsize=16)
        ax.set_ylabel(var[1].xlabel, fontsize=16)
    fig.tight_layout()

    if 'colorbar' in pargs:
        cbar_pad = 0.02
        cbar_width = 0.02
        if pargs['colorbar']['location'] == 'top':
            cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y1+cbar_pad,
                        ax.get_position().width,
                        cbar_width])
        elif pargs['colorbar']['location'] == 'bottom':
            cax = fig.add_axes([ax.get_position().x0,
                        ax.get_position().y0-cbar_pad-cbar_width,
                        ax.get_position().width,
                        cbar_width])
        elif pargs['colorbar']['location'] == 'right':
            cbar_width *=3/4
            cax = fig.add_axes([ax.get_position().x1+cbar_pad,
                        ax.get_position().y0,
                        cbar_width,
                        ax.get_position().height])
        elif pargs['colorbar']['location'] == 'left':
            cbar_width *=3/4
            cax = fig.add_axes([ax.get_position().x0-cbar_pad-cbar_width,
                        ax.get_position().y0,
                        cbar_width,
                        ax.get_position().height])
        cbar = fig.colorbar(im, ax=ax, cax=cax, **pargs['colorbar'])
        if 'cbar_ticklabels' in pargs:
            cbar.set_ticklabels(pargs['cbar_ticklabels'])
        cbar.ax.tick_params(axis='both', labelsize=16)
        if 'label' in pargs['colorbar']:
            cbar.set_label(label=pargs['colorbar']['label'], size=16)
        if 'cbar_ticksalignment' in pargs:
            if pargs['colorbar']['location'] == 'top':
                for xticks in cbar.ax.get_xticklabels():
                    xticks.set_horizontalalignment(pargs['cbar_ticksalignment'])
            else:
                for yticks in cbar.ax.get_yticklabels():
                    yticks.set_verticalalignment(pargs['cbar_ticksalignment'])
    if 'legend' in pargs:
        # if 'bbox_to_anchor' in pargs['legend']:
        #     ax.legend(pargs['legend']['labels'], fontsize=pargs['legend']['fsize'], loc=pargs['legend']['loc'], bbox_to_anchor=pargs['legend']['bbox_to_anchor'])
        # else:
        ax.legend(**pargs['legend'])
    
    if pargs['show']==True:
        p.show()
    if psave != None:
        plot_save(fig, var, params, psave)
    return 0

def plot_additional(ax, var, padd, pargs, label=None):
    if label != None:
        if label[0] == 'w':
            text = label[1:]
            color = 'white'
        else:
            text = label
            color = 'black'
        label_text = dict(x = 0.0175,
                                    y = 0.925,
                                    text = text,
                                    fsize = 16,
                                    color = color,
                                    transData = False)
        ax.text(label_text['x'], label_text['y'], label_text['text'], transform=ax.transAxes, fontsize=label_text['fsize'], fontweight='bold', color=label_text['color'])
    
    if 'ncurves' in padd:
        for i in range(padd['ncurves']):
            if 'label' not in padd['curves'][i]:
                padd['curves'][i]['label'] = None
            if 'alpha' not in padd['curves'][i]:
                padd['curves'][i]['alpha'] = 1
            if 'zorder' not in padd['curves'][i]:
                padd['curves'][i]['zorder'] = 2
            ax.plot(padd['curves'][i]['x'], padd['curves'][i]['y'], color=padd['curves'][i]['color'], dashes=padd['curves'][i]['dashes'], linewidth=padd['curves'][i]['linewidth'], label=padd['curves'][i]['label'], alpha=padd['curves'][i]['alpha'], zorder=padd['curves'][i]['zorder'])
    
    if 'nbackgrounds' in padd:
        for i in range(padd['nbackgrounds']):
            match padd['background'][i]['orientation']:
                case 'vertical':
                    ax.axvspan(padd['background'][i]['left'], padd['background'][i]['right'], facecolor=padd['background'][i]['color'], alpha=padd['background'][i]['alpha'])
                case 'horizontal':
                    ax.axhspan(padd['background'][i]['bottom'], padd['background'][i]['top'], facecolor=padd['background'][i]['color'], alpha=padd['background'][i]['alpha'])
    if 'ntext' in padd:
        for i in range(padd['ntext']):
            if padd['text'][i]['transData'] == True:
                if 'ylim' not in pargs:
                    pargs['ylim'] = [np.min(var[1].array)/var[1].norm, np.max(var[1].array)/var[1].norm]
                if 'xlim' not in pargs:
                    pargs['xlim'] = [np.min(var[0].array)/var[0].norm, np.max(var[0].array)/var[0].norm]
                p.setp(ax, xlim = pargs['xlim'], ylim = pargs['ylim'], xscale=pargs['xscale'], yscale=pargs['yscale'])

                trans = ax.transData.transform((padd['text'][i]['x'], padd['text'][i]['y']))
                (padd['text'][i]['x'], padd['text'][i]['y']) = ax.transAxes.inverted().transform(trans)
            
            ax.text(padd['text'][i]['x'], padd['text'][i]['y'], padd['text'][i]['text'], transform=ax.transAxes, fontsize=padd['text'][i]['fontsize'], fontweight='bold', color=padd['text'][i]['color'])
    
    if 'narrows' in padd:
        for i in range(padd['narrows']):
            if padd['arrows'][i]['transData'] == True:
                if 'ylim' not in pargs:
                    pargs['ylim'] = [np.min(var[1].array)/var[1].norm, np.max(var[1].array)/var[1].norm]
                if 'xlim' not in pargs:
                    pargs['xlim'] = [np.min(var[0].array)/var[0].norm, np.max(var[0].array)/var[0].norm]
                p.setp(ax, xlim = pargs['xlim'], ylim = pargs['ylim'], xscale=pargs['xscale'], yscale=pargs['yscale'])
                x0 = padd['arrows'][i]['x']
                y0 = padd['arrows'][i]['y']
                trans = ax.transData.transform((x0, y0))
                (padd['arrows'][i]['x'], padd['arrows'][i]['y']) = ax.transAxes.inverted().transform(trans)
                x1 = x0 + padd['arrows'][i]['dx']
                y1 = y0 + padd['arrows'][i]['dy']
                trans = ax.transData.transform((x1, y1))
                (x1, y1) = ax.transAxes.inverted().transform(trans)
                padd['arrows'][i]['dx'] = x1 - padd['arrows'][i]['x']
                padd['arrows'][i]['dy'] = y1 - padd['arrows'][i]['y']

            ax.arrow(padd['arrows'][i]['x'], padd['arrows'][i]['y'], padd['arrows'][i]['dx'], padd['arrows'][i]['dy'], color=padd['arrows'][i]['color'], width=padd['arrows'][i]['width'], transform=ax.transAxes, alpha=padd['arrows'][i]['alpha'], head_width=padd['arrows'][i]['head_width'], head_length=padd['arrows'][i]['head_length'])
    
    if 'nmarkers' in padd:
        for i in range(padd['nmarkers']):
            ax.scatter(padd['markers'][i]['x'], padd['markers'][i]['y'], s=padd['markers'][i]['size'], c=padd['markers'][i]['color'], marker=padd['markers'][i]['marker'])
    
    if 'area_under_curve' in padd:
        if padd['area_under_curve']['ncurves'] == 1:    
            if 'y2' in padd['area_under_curve']:
                ax.fill_between(x = var[0].array/var[0].norm, y1 = var[1].array/var[1].norm, y2=padd['area_under_curve']['y2'], color=padd['area_under_curve']['color'], alpha=padd['area_under_curve']['alpha'])
            else:
                ax.fill_between(x = var[0].array/var[0].norm, y1 = var[1].array/var[1].norm, color=padd['area_under_curve']['color'], alpha=padd['area_under_curve']['alpha'])
        else:
            match len(var[0].array.shape):
                case 1:
                    for i in range(padd['area_under_curve']['ncurves']):
                        ax.fill_between(x = var[0].array/var[0].norm, y1 = var[1].array[i]/var[1].norm, color=padd['area_under_curve']['color'][i], alpha=padd['area_under_curve']['alpha'][i])
                case _:
                    for i in range(padd['area_under_curve']['ncurves']):
                        ax.fill_between(x = var[0].array[i]/var[0].norm, y1 = var[1].array[i]/var[1].norm, color=padd['area_under_curve']['color'][i], alpha=padd['area_under_curve']['alpha'][i])
    return 0
