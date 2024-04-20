import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
import xarray as xr
import numpy as np
import datetime as datetime
import sys
import esm_tools
from esm_tools import *

# Viz params
n_ticks = 10
line_width = 3
fontsize = 12
MPa_per_mmh2o = 1e-5



def get_years_from_xarr(xarr):
    years = [xarr.time[i].item().year for i in range(len(xarr.time))]
    return years

def getNBase(xarr):
    nyears = len(np.unique(get_years_from_xarr(xarr.time)))
    nbase = max(nyears // n_ticks, 2)
    return(nbase)


def get_n_subplots(n_pfts): 
 
    if (n_pfts % 2 == 0) | (n_pfts == 1): 
        n_subplots = n_pfts 
    else: 
        n_subplots = n_pfts + 1

    if n_subplots == 1:
        ncol = 1
        nrow = 1

    else:
        ncol = 2
        nrow = n_subplots / ncol

    return (ncol,int(nrow))

def plot_multi_panel(df, x_col, y_cols, figsize=(6, 8),
                     save_fig = False,
                     output_path_for_case = None):
    """
    Plots multiple y-columns against one x-column in a multi-panel figure.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - x_col (str): Name of the column for the x-axis.
    - y_cols (list of str): List of column names for the y-axes.
    - figsize (tuple, optional): Figure size. Defaults to (6, 8).
    """
    # Create subplots
    fig, axs = plt.subplots(nrows=len(y_cols), figsize=figsize, sharex=True)
    
    # If only one y_col is provided, axs is not a list; make it one for consistent indexing
    if len(y_cols) == 1:
        axs = [axs]
    
    # Plot each y_col
    for i, ycol in enumerate(y_cols):
        axs[i].scatter(df[x_col], df[ycol], label=ycol)
        axs[i].legend(loc='upper right')
        axs[i].set_ylabel(ycol)
    
    axs[-1].set_xlabel(x_col)
    if save_fig == True:
         fig_file_name = "ensemble_fig_" + x_col + ".png"
         plt.savefig(os.path.join(output_path_for_case,fig_file_name))
    plt.tight_layout()
    plt.show()

   


def plot_array(xarr,xds,n_pfts,pft_colors,pft_names,title,ylabel,output_path,conversion = 1,subplots = False, getData = False, dbh_min = None):
    
    xarr = xarr * conversion
    nbase = getNBase(xarr)

    #prep data if it has multiplexed dimensions
    if xarr.dims == ('time', 'fates_levscpf'):
        xarr = esm_tools.scpf_to_scls_by_pft(xarr, xds)
        if dbh_min != None:
            xarr = xarr.sel(fates_levscls = slice(dbh_min,None))
        xarr = xarr.sum(dim="fates_levscls")
    
    if subplots == False:

        fig, ax = plt.subplots(figsize = (5,5))
        
        if xarr.dims == ('time', 'fates_levpft'):

            for p in range(n_pfts):
                xarr.isel(fates_levpft=p).plot(x = "time",
                color = pft_colors[p],lw = line_width,add_legend = True, label = pft_names[p])
                    
            plt.legend()

        if xarr.dims == ('time',):

            xarr.plot(x = "time", lw = line_width)
        

        #ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        #ax.xaxis.set_major_locator(mdates.YearLocator(base=nbase))
        ax.set_ylabel(ylabel,fontsize = fontsize)
        ax.title.set_text(title)


    if subplots == True:
        
        ncol,nrow = get_n_subplots(n_pfts)
        fig, axes = plt.subplots(ncols=ncol,nrows=nrow,figsize=(7,7))
        
        for p,ax in zip(range(n_pfts),axes.ravel()):
            xarr.isel(fates_levpft = p).plot(ax = ax, x = "time", color = pft_colors[p],
                                             lw = line_width)
            ax.set_title('{}'.format(pft_names[p]))
            ax.set_ylabel(ylabel,fontsize = fontsize)
            #ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            #ax.xaxis.set_major_locator(mdates.YearLocator(base=nbase*2))


            if (pft_names[p] == "shrub") & (title == "Stem Density"):
                ax.set_yscale("log")

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6,wspace=0.6)
        fig.suptitle(title, fontsize=fontsize)

    #plt.savefig(output_path + "/" + title.replace(" ","") + ".pdf")
    #plt.savefig(output_path + "/" + title.replace(" ","") + ".png")
    
    if getData == True:
        return(xarr)

def plot_smp(xarr,xds,output_path,depths = [4,7,8],return_means = False):
    title = "SMP"
    #nbase = getNBase(xarr)
 
    fig, ax = plt.subplots(figsize = (7,7))

    mean_smp = []
    for d in depths:
        smp = xarr.isel(levgrnd = d) * MPa_per_mmh2o
        smp.plot(label = xds.levgrnd.values[d])
        plt.legend()

    #ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    #ax.xaxis.set_major_locator(mdates.YearLocator(base=nbase))
    ax.set_ylabel("SMP [MPa]")
    ax.title.set_text("SMP [MPa]")

    #plt.savefig(output_path + "/" + title.replace(" ","") + ".pdf")
    #plt.savefig(output_path + "/" + title.replace(" ","") + ".png")


def plot_ap(xarr, xds, sup_title, ylabel, output_path):
    nbase = getNBase(xarr) * 2
    n_age = len(xds.fates_levage)
    ncol,nrow = get_n_subplots(n_age)
    fig, axes = plt.subplots(ncols=ncol,nrows=nrow,figsize=(10,10))

    for age,ax in zip(range(n_age),axes.ravel()):
        xds.FATES_PATCHAREA_AP.isel(fates_levage = age).plot(ax = ax)
        ax.set_title('{} yr old patches'.format(xds.fates_levage.values[age]))
        ax.set_ylabel(ylabel,fontsize = int(fontsize * 0.75))
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=nbase))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6,wspace=0.6)
    fig.suptitle(sup_title, fontsize=fontsize)

    #plt.savefig(output_path + "/" + sup_title.replace(" ","") + ".pdf")
    #plt.savefig(output_path + "/" + sup_title.replace(" ","") + ".png")

def plot_ba(ds,n_pfts,pft_names):
        #disentangle the multiplexed size class X pft dimension
        basal_area = esm_tools.scpf_to_scls_by_pft(ds.FATES_BASALAREA_SZPF, ds)

        #sum across size classes to get pft-level ba
        basal_area_pf = basal_area.sum(dim="fates_levscls")

        #plot pft-level basal area over time
        for p in range(n_pfts):
            ba_per_ha = basal_area_pf.isel(fates_levpft=p) * m2_per_ha
            ba_per_ha.plot(x = "time", color = pft_colors[p],lw = 5,label = pft_names[p])

        plt.title("Basal Area")
        plt.legend()
        plt.ylabel('BA [m-2 ha-1]', fontsize=12)
        
def plot_size_class_distribution(ds,n_pfts,pft_colors,pft_names,variable_type = "BA", final_timestep = True,
                                 specific_time_step = None,
                                 conversion = 1,
                                 dbh_min = None,pft_specific = True):
    
        
    
    if variable_type == "BA":
        xarr = esm_tools.scpf_to_scls_by_pft(ds.FATES_BASALAREA_SZPF, ds)
    else:
        xarr = esm_tools.scpf_to_scls_by_pft(ds.FATES_NPLANT_SZPF, ds)
    
    if dbh_min != None:
        xarr = xarr.sel(fates_levscls = slice(dbh_min,None))
     
    if final_timestep == True:
        xarr = xarr.isel(time = -1)
    if specific_time_step != None:
        xarr = xarr.sel(time = specific_time_step)
    
    xarr = xarr * conversion
    
 
    if pft_specific == False:
        xarr = xarr.isel(fates_levpft = np.array([0,1,2,4]))       


    ncol,nrow = get_n_subplots(n_pfts)
    fig, axes = plt.subplots(ncols=ncol,nrows=nrow,figsize=(10,5))
    
    for p,ax in zip(range(n_pfts),axes.ravel()):
        
        xarr_pf = xarr.isel(fates_levpft=p)
        
        xarr_pf.plot(ax=ax, x = "fates_levscls", marker = "o", lw = 0, 
                  color = pft_colors[p], markersize = 5)
        ax.title.set_text('{}'.format(pft_names[p]))
        x_ticks = ds.fates_levscls.values   # Adjust the range and step as needed
        ax.set_xticks(x_ticks)
        
        if variable_type == "BA":
            ax.set_ylabel('BA [m-2 ha-1]')
        if variable_type == "Stem density":
            ax.set_ylabel('Stem density [N ha-1]')

    plt.tight_layout()
    
def plot_appf(xarr, xds, n_pfts, sup_title, ylabel, output_path):

    xarr = appf_to_ap_by_pft(xarr, xds)

    n_age = len(xds.fates_levage)

    ncol,nrow = get_n_subplots(n_age)

    #nbase = getNBase(xarr) * 2

    fig, axes = plt.subplots(ncols=ncol,nrows=nrow,figsize=(12,10))

    for age,ax in zip(range(n_age),axes.ravel()):

         cca = xarr.isel(fates_levage = age) / xds.FATES_PATCHAREA_AP.isel(fates_levage = age)

         for p in range(n_pfts):
             cca.isel(fates_levpft=p).plot(x = "time",
                      color = pft_colors[p],lw = 3,add_legend = True,
                      label = pft_names[p], ax = ax)

             #plt.legend()
         ax.set_title('{} yr old patches'.format(xds.fates_levage.values[age]))
         ax.set_ylabel(ylabel,fontsize = int(12 * 0.75))
         ax.xaxis.set_major_formatter(DateFormatter('%Y'))
         #ax.xaxis.set_major_locator(mdates.YearLocator(base=nbase))

    plt.tight_layout()
    plt.subplots_adjust(hspace=1,wspace=0.2)
    fig.suptitle(sup_title, fontsize=12,y=0.99)
    
def cca_by_patch_age(ds,n_pfts,pft_names,pft_colors,canopy_crown_area = False):
    
    if canopy_crown_area == False:
        xarr = appf_to_ap_by_pft(ds.FATES_CROWNAREA_APPF, ds)
        title = "Crown area"
    else:
        xarr = appf_to_ap_by_pft(ds.FATES_CANOPYCROWNAREA_APPF, ds)
        title = "Canopy crown area"
    
    # Normalize by patch area
    xarr = xarr / ds.FATES_PATCHAREA_AP

    for p in range(n_pfts):
        xarr.isel(fates_levpft = p).mean(dim = "time").\
        plot(x = "fates_levage",color = pft_colors[p], linewidth = 3, marker = "o")

    plt.title(title)
    plt.xlabel("Patch age bin (yrs)")
    plt.ylabel("Crown area [m2 m-2]")
    
def plot_area_weighted_fire_intensity(ds):
    aw_fi = esm_tools.get_awfi(ds)
    aw_fi.plot(marker = "o",linewidth = 0.5)
    plt.ylabel("Fire line intensity [kW m-1]")
    title = "Fire Intensity"
    plt.title(title)
    #plt.savefig(output_path + "/" + case + "_" + title.replace(" ","-") + ".png")
    #plt.clf()

def plot_awfi_hist(ds,start_date=None,end_date=None):
    aw_fi = esm_tools.get_awfi(ds)
    aw_fi = aw_fi.sel(time = slice(start_date,end_date))
    #print(aw_fi.values)
    plt.hist(aw_fi.values, bins=20, edgecolor='black')
    plt.xlabel('Area-weighted burn intensity [kW m-1]')
    plt.ylabel('Frequency')
    plt.title("Area-weighted burn intensity")

    
def plot_mean_annual_burn_frac(ds,start_date=None,end_date=None):
    burnfrac = ds.FATES_BURNFRAC  * s_per_yr
    total_mean_annual_burnfrac = esm_tools.get_mean_annual_burn_frac(ds,start_date,end_date)

    annual_mean_burnfrac = burnfrac.groupby('time.year').mean(dim='time').values
    title = f"Mean annual burn fraction: {np.round(total_mean_annual_burnfrac,3)}"
    
    # Create a histogram of the distribution of annual means
    plt.hist(annual_mean_burnfrac, bins=20, edgecolor='black')
    plt.xlabel('Annual burn fraction')
    plt.ylabel('Frequency')
    plt.title(title)
    x_ticks = np.linspace(min(annual_mean_burnfrac), max(annual_mean_burnfrac), 5)
    plt.xticks(x_ticks)
