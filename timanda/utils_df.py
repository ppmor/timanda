import allantools
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import os
from report import Report
import uuid


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
    if mjd_shift != 0:
        df[mjd] = df[mjd] + mjd_shift
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
    (taus, adevs, errors, ns) = allantools.adev(
        data,
        rate=1/300,
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
    f_name = gen_file_name()
    plt.savefig(f_name)
    plt.close('all')
    rep.dodaj_wykres(f_name)


def add_df_to_rep(df, param, rep, text=None):
    if text:
        rep.dodaj_tekst(text)
    plot_df(df, param, rep)
    plot_df_allan(df, param, rep)
