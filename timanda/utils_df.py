import allantools
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import os
from report import Report
import uuid
from scipy.stats import linregress
import pandas as pd


def mjd_to_datetime(mjd):
    return Time(mjd, format='mjd').to_datetime()


def noise_analysis_df(df, mjd='mjd', val='val'):
    return 0


def resample_df(
    df,
    mjd='mjd',
    period_str='5T',
    method='mean',
    mjd_shift=0,
):
    df['datetime'] = mjd_to_datetime(df[mjd])
    df.set_index('datetime', inplace=True)
    if method == 'mean':
        df = df.resample(period_str).mean()
    if method == 'first':
        df = df.resample(period_str).first()
    if method == 'slope':
        df = df.resample(period_str).apply(get_slope)
    if mjd_shift != 0:
        df[mjd] = df[mjd] + mjd_shift
    df['mjd'] = df.index.map(lambda x: Time(x).mjd)
    return df


def rm_wykres_files():
    files = os.listdir()
    for file in files:
        if file.startswith('wykres_') and file.endswith('.png'):
            os.remove(file)


def start_report(report_file_name="rep.pdf"):
    rm_wykres_files()
    return Report(report_file_name)


def stop_report(rep):
    rep.zapisz()
    rm_wykres_files()


def gen_file_name():
    return f"wykres_{uuid.uuid4()}.png"


def plot_df(df, param, rep):
    plt.scatter(df['mjd'], df[param], marker='o', s=1.5)
    plt.xlabel('mjd')
    plt.ylabel(param)
    plt.title(param + ' vs mjd')
    plt.grid(True)
    f_name = gen_file_name()
    plt.savefig(f_name)
    plt.close('all')
    rep.dodaj_wykres(f_name)


def plot_df_allan(df, param, rep):
    t = np.power(10, np.arange(1, int(np.log10(50*24*60*60))+0.1, 0.1))
    data = df[param].to_numpy()
    mjd_max = df['mjd'].max()
    mjd_min = df['mjd'].min()
    mjd_count = df['mjd'].count()
    rate = 1 / ((mjd_max-mjd_min)*24*60*60 / mjd_count)
    (taus, adevs, errors, ns) = allantools.adev(
        data,
        rate=rate,
        data_type='freq',
        taus=t
    )
    plt.errorbar(taus, adevs, yerr=errors, fmt='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('ADEV')
    plt.title(param)
    plt.grid(True)
    if rep:
        f_name = gen_file_name()
        plt.savefig(f_name)
        plt.close('all')
        rep.dodaj_wykres(f_name)
    else:
        plt.plot()
        plt.show()


def add_df_to_rep(df, param, rep, text=None):
    if text:
        rep.dodaj_tekst(text)
    plot_df(df, param, rep)
    plot_df_allan(df, param, None)


def rm3sigma_df(df, param='frac_freq'):
    init_len = df.shape[0]
    m = df[param].mean()
    std = df[param].std()
    df = df[abs(df[param]-m) < 3*std]
    if init_len > df.shape[0]:
        df = rm3sigma_df(df, param)
    return df


def get_slope(group):
    if len(group) < 2:
        return np.nan
    seconds_since_epoch = (
        (group.index - pd.Timestamp('1970-01-01')).total_seconds() * 1e9
    )
    slope, intercept, r_value, p_value, str_err = linregress(
        seconds_since_epoch, group
    )
    return slope
