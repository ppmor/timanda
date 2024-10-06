from astropy.time import Time
from scipy.stats import linregress
import decimal as dec  # TODO: remove after removing alphanorm
from decimal import Decimal as D
from decimal import getcontext
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
import allantools as al
from astropy.convolution import Gaussian1DKernel, convolve

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s - timanda - %(message)s'
)

class TSerie:
    def __init__(self, label='', mjd=[], val=[], pps=None):
        self.label = label
        self.mjd_tab = np.array(mjd)
        self.val_tab = np.array(val)
        if pps is None:
            self.pps_tab = np.array([1]*len(mjd)) # points per sample, used when resampling
        else:
            self.pps_tab = np.array(pps)
        self.len = len(self.mjd_tab)
        self.calc_tab()

    def __str__(self):
        if len(self.mjd_tab) == 0:
            s = '\tEmpty'
        else:
            self.calc_tab()
            s = 'TSerie:\tlabel: %s\tlength: %d\tlen_mjd: %.6f\n' % (
                self.label,
                self.len,
                self.len_mjd
            )
            if self.len <= 10:
                for i in range(0, self.len):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i],
                    )
            else:
                for i in range(0, 5):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i]
                    )
                s = s+'\t...\n'
                for i in range(self.len-5, self.len):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i],
                    )
        return s

    def __add__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab + b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('Adding TSerie to TSerie is  not supported yet')
            return None
        else:
            return None

    def __sub__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab - b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print(' TSerie - TSerie is  not supported yet')
            return None
        else:
            return None

    def __mul__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab * b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('TSerie * TSerie is  not supported yet')
            return None
        else:
            return None

    def __truediv__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab / b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('TSerie / TSerie is  not supported yet')
            return None
        else:
            return None

    def calc_tab(self):  # if not empty
        self.len = len(self.mjd_tab)
        if self.len > 0:
            self.isempty = 0
            self.mjd_start = self.mjd_tab[0]
            self.mjd_stop = self.mjd_tab[-1]
            self.mean = np.mean(self.val_tab)
        else:
            self.isempty = 1
            self.mjd_start = 0
            self.mjd_stop = 0
        mjd_s = 24*60*60
        self.s_tab = (self.mjd_tab - self.mjd_start)*mjd_s
        self.t_type = 't_type'
        self.len_mjd = self.mjd_stop-self.mjd_start
        self.len_s = self.len_mjd*mjd_s

    def len(self):
        return len(self.mjd_tab)

    def cp(self):
        out = TSerie(label=self.label+'_cp',
                     mjd=self.mjd_tab,
                     val=self.val_tab)
        return out

    def mean_use_pps(self, decimal=False, decimal_out=False):
        if len(self.val_tab) > 0:
            if decimal:
                getcontext().prec = 21
                d_pps_tab = [D(str(x)) for x in self.pps_tab]
                d_val_tab = [D(str(x)) for x in self.val_tab]
                d_pps_val_tab = [x*y for x, y in zip(d_pps_tab, d_val_tab)]
                d_sum_points = sum(d_pps_tab)
                d_mean = sum(d_pps_val_tab)/d_sum_points
                if decimal_out:
                    return d_mean, d_sum_points
                else:
                    return float(d_mean), int(d_sum_points)
            else:
                pps_val_tab = self.pps_tab*self.val_tab
                sum_points = sum(pps_val_tab)
                mean_out = sum(pps_val_tab)/float(sum_points)
                return mean_out, int(sum_points)
        else:
            return None

    def mean(self, decimal=False, decimal_out=False, use_pps=False):
        if use_pps:
            return self.mean_use_pps(decimal=decimal, decimal_out=decimal_out)
        if len(self.val_tab) > 0:
            if decimal:
                getcontext().prec = 21
                darr = [D(x) for x in self.val_tab]
                if decimal_out:
                    return sum(darr)/len(darr)
                else:
                    return float(sum(darr)/len(darr))
            else:
                return np.mean(self.val_tab)
        else:
            return None
        
    def max_val(self):
        return np.max(self.val_tab)

    def rm_dc(self):
        self.val_tab = self.val_tab - self.mean

    def rm_drift(self):
        try:
            fit = np.polyfit(self.mjd_tab, self.val_tab, 1)
            self.val_tab = (
                self.val_tab - (self.mjd_tab*fit[0]+fit[1])
            )
        except:  # TODO: add exception name here
            pass
            # print('rm drift problem')

    def split(self, min_gap_s=8):
        if self.len == 0:
            return []
        out_tab = []
        tab_i = []
        tab_i.append(0)
        for i in range(0, len(self.s_tab)-1):
            if self.s_tab[i+1]-self.s_tab[i] > min_gap_s:
                tab_i.append(i+1)
        tab_i.append(len(self.s_tab)-1)
        for j in range(0, len(tab_i)-1):
            out_tab.append(TSerie(
                mjd=self.mjd_tab[tab_i[j]:tab_i[j+1]],
                val=self.val_tab[tab_i[j]:tab_i[j+1]],
                pps=self.pps_tab[tab_i[j]:tab_i[j+1]],
            ))
        return out_tab

    def append(self, mjd, val, pps=None):
        self.mjd_tab = np.append(self.mjd_tab, mjd)
        self.val_tab = np.append(self.val_tab, val)
        if pps:
            self.pps_tab = np.append(self.pps_tab, pps)
        else:
            self.pps_tab = np.append(self.pps_tab, 1)

    def last(self):
        out = TSerie()
        out.append(self.mjd_tab[-1], self.val_tab[-1], self.pps_tab[-1])
        return out

    def mjd2index(self, mjd, init_index=None):
        """Returns index of tab corresponding to mjd
        in: mjd, init_index - initial index for searching
        out: index of the nearest mjd <= input_mjd
             None if mjd is out of table
        """
        if mjd < self.mjd_start or mjd > self.mjd_stop:
            return None
        if init_index is None:
            if self.len_mjd == 0:
                return None
            N = int((self.len/self.len_mjd)*(mjd-self.mjd_start))
        else:
            N = init_index
        if N < 0:
            N = 0
        if N >= self.len:
            N = self.len-1
        while 1:
            if self.mjd_tab[N] > mjd:
                N = N-1
            else:
                if N+1 >= self.len or self.mjd_tab[N+1] > mjd:
                    return N
                else:
                    N = N+1

    def mjd2val(self, mjd, init_index=None):
        return self.val_tab[self.mjd2index(mjd, init_index=init_index)]

    def getrange(self, fmjd, tmjd):
        if (fmjd > self.mjd_stop or tmjd < self.mjd_start):
            return None
        if fmjd < self.mjd_start:
            fmjd = self.mjd_start
        if tmjd > self.mjd_stop:
            tmjd = self.mjd_stop
        fN = self.mjd2index(fmjd)
        tN = self.mjd2index(tmjd)
        if tN is None:
            return None
        s = TSerie(
            mjd=self.mjd_tab[fN:tN+1],
            val=self.val_tab[fN:tN+1]
        )
        if s.mjd_tab[0] < fmjd:
            s.rm_first(1)
        if len(s.mjd_tab) == 0:
            return None
        if s.mjd_tab[-1] > tmjd:
            s.rm_last(1)
        return s

    def rmrange(self, fmjd, tmjd):
        if (fmjd > self.mjd_stop or tmjd < self.mjd_start):
            return (self, 0)
        if (fmjd <= self.mjd_start and tmjd >= self.mjd_stop):
            return (None, 1)
        fN = self.mjd2index(fmjd)
        tN = self.mjd2index(tmjd)
        if (fmjd <= self.mjd_start and tmjd < self.mjd_stop):
            return (TSerie(
                mjd=self.mjd_tab[(tN+1):],
                val=self.val_tab[(tN+1):],
                pps=self.pps_tab[(tN+1):],
                ),
                2
            )
        if (fmjd > self.mjd_start and tmjd >= self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            return (TSerie(
                mjd=self.mjd_tab[:fN],
                val=self.val_tab[:fN],
                pps=self.pps_tab[:fN],
                ),
                3)
        if (fmjd > self.mjd_start and tmjd < self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            left = TSerie(
                mjd=self.mjd_tab[:fN],
                val=self.val_tab[:fN],
                pps=self.pps_tab[:fN],
            )
            right = TSerie(
                mjd=self.mjd_tab[(tN+1):],
                val=self.val_tab[(tN+1):],
                pps=self.pps_tab[(tN+1):],
            )
            return ([left, right], 4)
    
    def rm_indexes(self, indexes):
        self.mjd_tab = np.delete(self.mjd_tab, indexes)
        self.val_tab = np.delete(self.val_tab, indexes)
        self.pps_tab = np.delete(self.pps_tab, indexes)
        self.len = len(self.mjd_tab)

    def rm_outlayers_singledelta(self, max_delta):
        self.calc_tab()
        for i in range(1, self.len):
            if abs(self.val_tab[i] - self.val_tab[i-1]) > max_delta:
                print('outlayer detected')
                self.val_tab[i] = self.val_tab[i-1]

    def rmOutlayersOfTarget(self, target, maxDifference, get_indexes_only=False):
        indexes_to_delete = []
        i = 0
        self.len=len(self.mjd_tab)
        while i < self.len:
            if abs(self.val_tab[i]-target) > maxDifference:
                indexes_to_delete.append(i)
            i = i+1
        if not get_indexes_only:
            self.rm_indexes(indexes_to_delete)
        return indexes_to_delete
    
    def rm_value(self, value, get_indexes_only=False):
        indexes_to_delete = []
        i = 0
        while i < self.len:
            if self.val_tab[i] == value:
                indexes_to_delete.append(i)
            i = i+1
        if not get_indexes_only:
            self.rm_indexes(indexes_to_delete)
        return indexes_to_delete

    def time_shift(self, sec):
        self.mjd_tab = self.mjd_tab+sec/(24*60*60)
        self.calc_tab()

    def show(self):
        N = 5
        if len(self.mjd_tab) > 2*N:
            for x in range(0, 5):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
            print('...')
            for x in range(self.length-6, self.length-1):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
        elif len(self.mjd_tab) > 0:
            for x in range(0, 2*N-1):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
        else:
            print('Empty')

    def plot(self):
        plt.figure()
        plt.title(self.label)
        # plt.plot(self.mjd_tab, self.val_tab)
        plt.scatter(self.mjd_tab, self.val_tab*0, s=700, marker="|")
        plt.show()

    def plot_pqg(self, minusmjd):
        w = pg.plot(self.mjd_tab-minusmjd, self.val_tab)
        w.setBackground('w')
        w.setTitle(self.label)
        w.show()
        return w

    def plot_pqg_widget(self, minusmjd=0):
        self.widget = pg.PlotWidget()
        self.widget.plot(self.mjd_tab-minusmjd, self.val_tab)
        self.widget.setBackground('w')
        self.widget.setTitle(self.label)
        return self.widget

    def plot_nice(self, fig=0, ax_site='left'):
        if fig == 0:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.mjd_tab, self.val_tab)
        plt.show()
        return fig, ax

    def plot_allan_pqg(self, atom='88Sr'):
        if atom == '88Sr':
            fabs = 400e12
        w = pg.PlotWidget()
        t = np.power(10, np.arange(1, int(np.log10(self.len_s))+0.1, 0.1))
        # t=np.logspace(np.log10(20),np.log10(self.len_s),50)
        y = self.val_tab
        r = self.len_s/self.len
        (t2, ad, ade, adn) = al.adev(y, rate=r, data_type="freq", taus=t)
        w.plot(t2, ad/fabs, symbol='o')
        w.showGrid(True, True)
        w.setLogMode(True, True)
        w.setBackground('w')
        w.setTitle(self.label+'_ADEV')
        return w

    def plot_allan(self, atom='88Sr', fabs=None, out_file_name=None):
        if atom == '88Sr':
            fabs = 429228066418012
        t = np.power(10, np.arange(1, int(np.log10(self.len_s))+0.1, 0.1))
        y = self.val_tab/fabs
        r = self.len_s/self.len
        a = al.Dataset(data=y, rate=r, data_type="freq", taus=t)
        a.compute('adev')
        b = al.Plot()
        b.plot(a, errorbars=True, grid=True)
        if out_file_name:
            plt.savefig(out_file_name)
            plt.close()
        else:
            b.show()

    def scatter(self):
        plt.scatter(self.mjd_tab, self.val_tab)
        plt.show()

    def rm_first(self, n=1):
        self.val_tab = np.delete(self.val_tab, np.s_[0:n], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[0:n], None)
        self.pps_tab = np.delete(self.pps_tab, np.s_[0:n], None)
        self.calc_tab()

    def rm_last(self, n=1):
        self.val_tab = np.delete(self.val_tab, np.s_[-n:], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[-n:], None)
        self.pps_tab = np.delete(self.pps_tab, np.s_[-n:], None)
        self.calc_tab()

    def gauss_filter(self, stddev=50):
        g = Gaussian1DKernel(stddev=stddev)
        self.val_tab = convolve(self.val_tab, g)
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)

    def high_gauss_filter(self, stddev=50, rm_dc=True):
        if not rm_dc:
            dc = self.mean
        g = Gaussian1DKernel(stddev=stddev)
        tmp = convolve(self.val_tab, g)
        self.val_tab = self.val_tab - tmp
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)
        if not rm_dc:
            self.val_tab = self.val_tab + dc

    def toMTSerie(self):
        return MTSerie(TSerie=self)
    
    def first_mjd(self):
        return self.mjd_tab[0]
    
    def last_mjd(self):
        return self.mjd_tab[-1]
    
    def time_diff_to_freq_diff(self, fill_last_point=True):
        t_mjd=list()
        t_val=list()
        for i in range(0, len(self.mjd_tab)-1):
            delta_mjd_s = (self.mjd_tab[i+1]-self.mjd_tab[i])*(24*60*60)
            f = (self.val_tab[i+1]-self.val_tab[i])/delta_mjd_s
            t_mjd.append(self.mjd_tab[i])
            t_val.append(f)
        if fill_last_point:
            t_val.append(t_mjd[-1])
            t_mjd.append(self.mjd_tab[-1])
        self.mjd_tab=np.array(t_mjd)
        self.val_tab=np.array(t_val)
    
    def add_sin(self, amplitude=0, omega=0):
        mjd2s = 24*60*60
        for i,v in enumerate(self.val_tab):
            t = (self.mjd_tab[i])*mjd2s
            self.val_tab[i] += amplitude*np.sin(omega*t)
             

class MTSerie:
    def __init__(self, label='', TSerie=None, color='green', txtFileName=None,
                 plot_label='', plot_ref_val=0):
        self.label = label
        self.plot_label = plot_label
        self.plot_ref_val = plot_ref_val
        self.dtab = []
        self.color = color
        if txtFileName is not None:
            self.importFromTxtFile(txtFileName)
        if TSerie is not None:
            self.add_TSerie(TSerie)

    def importFromTxtFile(self, fileName):
        f = open(fileName, 'r')
        mjd_t = []
        val_t = []
        for line in f:
            if line[0] != '#':
                mjd_t.append(float(line.split()[0]))
                val_t.append(float(line.split()[1]))
        f.close()
        self.add_TSerie(TSerie(mjd=mjd_t, val=val_t))
        self.split()

    def __str__(self):
        s = f"MTSerie {self.label}:\n"
        for x in self.dtab:
            s = s+x.__str__()
        return s

    def __iadd__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i]+b
        return self

    def __add__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]+b)
        return out

    def __sub__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]-b)
        elif isinstance(b, (MTSerie)):
            ts = TSerie()
            tp1 = self.getTimePeriods()
            tp2 = b.getTimePeriods()
            tc = tp1.commonPart(tp2)
            for per in tc.periods:
                mjds = np.arange(per.start, per.stop, 1/(60*60*24))
                for mjd in mjds:
                    ts.append(mjd, self.mjd2val(mjd)-b.mjd2val(mjd))
            out = MTSerie()
            out.add_TSerie(TSerie(mjd=ts.mjd_tab, val=ts.val_tab))
            out.split()
        return out

    def __imul__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i] * b
        return self

    def __mul__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]*b)
        return out

    def __idiv__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i] / b
        return self

    def isempty(self):
        for x in self.dtab:
            if x.len > 0:
                return False
        return True

    def calc_tabs(self):
        for x in self.dtab:
            x.calc_tab()

    def rmemptyseries(self):
        self.calc_tabs()
        self.dtab = [x for x in self.dtab if x.isempty == 0]

    def getTimePeriods(self):
        out = TimePeriods()
        for x in self.dtab:
            if len(x.mjd_tab) > 0:
                out.appendPeriod(TimePeriod(x.mjd_tab[0], x.mjd_tab[-1]))
        return out

    def getTotalTimeWithoutGaps(self):
        tp = self.getTimePeriods()
        if tp:
            return tp.totalTimeWithoutGaps()
        else:
            return None

    def alphnorm(self, atom='88Sr'):
        d = {
            # ref: Safranova RevModPhys 2018  page 8
            '88Sr': {'fabs': dec.Decimal('429228066418000'), 'K': 1.06},
            '87Sr': {'fabs': dec.Decimal('429228004229873'), 'K': 1.06},
            '171Yb': {'fabs': dec.Decimal('518295836590865'), 'K': 1.31},
            '171Yb+': {'fabs': dec.Decimal('688358979309307'), 'K': 2.03},
            # dla E2
            # 1-5.95 dla E3
        }
        for x in d.keys():
            d[x]['factor'] = float(
                d[x]['fabs']*dec.Decimal(d[x]['K'])*dec.Decimal('1e-18')
            )
        if atom in d.keys():
            k = d[atom]['factor']
        else:
            return None
        for x in self.dtab:
            x.val_tab = x.val_tab / k

    def rm_drift_each(self):
        for x in self.dtab:
            x.rm_drift()

    def add_TSerie(self, ser):
        self.dtab.append(ser)

    def add_mjdf_data(self, mjd, f):
        tmp = TSerie(mjd=mjd, val=f)
        self.dtab.append(tmp)

    def add_mjdf_from_file(self, file_name):
        raw = np.load(file_name, allow_pickle=True)
        self.add_mjdf_data(raw[:, 0], raw[:, 1])

    def add_mjdf_from_datfile(self, file_name):
        raw = np.loadtxt(file_name)
        self.add_mjdf_data(raw[:, 0], raw[:, 1])

    def plot(self, color='', show=1, ax=None, zorder=1, marker=".", linestyle='none',
             nolabels=False):
        for x in self.dtab:
            if color == '':
                color = self.color
            if ax is None:
                plt.plot(x.mjd_tab, x.val_tab-self.plot_ref_val,
                         color=color, marker=marker,
                         linestyle=linestyle, zorder=zorder)
            else:
                ax.plot(x.mjd_tab, x.val_tab-self.plot_ref_val,
                        color=color, marker=marker,
                        linestyle=linestyle, zorder=zorder)
            if not nolabels:
                plt.ylabel(self.plot_label)
        if show == 1:
            plt.show()

    def hist(self, bins=10):
        v = self.val_tab()
        plt.hist(v, bins=bins)

    def plot_pqg_widget(self, minusmjd=0, widget=None):
        if widget is None:
            self.widget = pg.PlotWidget()
        else:
            self.widget = widget
        for x in self.dtab:
            if len(x.mjd_tab) > 0:
                print("---------")
                print(x)
                self.widget.plot(x.mjd_tab-minusmjd, x.val_tab)
        if widget is None:
            self.widget.setBackground('w')
            self.widget.setTitle(self.label)
        return self.widget

    def plot_allan(self, atom=None, ref_val=None, rate=1, taus=None):
        if ref_val:
            ref = ref_val
        else:
            ref = 1
        if atom == '88Sr':
            ref_val = 429228066418012.0
        y = self.val_tab()/ref
        # y = y.flatten()
        # print('y: ', y)
        if taus is None:
            taus = np.power(10, np.arange(0, int(np.log10(len(y)/rate))+0.1, 0.1))
        a = al.Dataset(data=y, rate=rate, data_type="freq", taus=taus)
        a.compute('adev')
        b = al.Plot()
        b.plot(a, errorbars=True, grid=True)
        b.show()

    def sew(self, grid_s=1):
        g = grid_s/(24*60*60)
        out = []
        for x in self.dtab:
            for mjd in np.arange(x.mjd_tab[0], x.mjd_tab[-1], g):
                out.append(x.mjd2val(mjd))
        return np.array(out)

    def split(self, min_gap_s=8):
        tmp_tab = []
        for a in self.dtab:
            spl = a.split(min_gap_s)
            for s in spl:
                tmp_tab.append(s)
        self.dtab = tmp_tab

    def rm_dc_each(self):
        for x in self.dtab:
            x.rm_dc()

    def rm_dc(self):
        mean = self.mean()
        for i in range(0, len(self.dtab)):
            self.dtab[i] = self.dtab[i]-mean

    def high_gauss_filter_each(self, stddev=50, rm_dc=True):
        i = 0
        for x in self.dtab:
            i = i+1
            # print('filter '+str(i))
            if x.len < stddev*10:
                pass
                # print('to short')
                # self.dtab.remove(x)
            else:
                x.high_gauss_filter(stddev=stddev, rm_dc=rm_dc)

    def time_shift_each(self, sec):
        for x in self.dtab:
            x.time_shift(sec)

    def mjd2tabNo(self, mjd):
        for num in range(0, len(self.dtab)):
            if (
                len(self.dtab[num].mjd_tab) > 0 and
                mjd < self.dtab[num].mjd_tab[0]
            ):
                return num-1
        return len(self.dtab)-1

    def mjd2tabNoandindex(self, mjd):
        tabNo = self.mjd2tabNo(mjd)
        if tabNo == -1:
            index = None
        else:
            index = self.dtab[tabNo].mjd2index(mjd)
        return (tabNo, index)

    def mjd2val(self, mjd, mode=0):
        (t, i) = self.mjd2tabNoandindex(mjd)
        if i is None:
            return None
        else:
            if mode == 1:
                v = self.dtab[t].val_tab[i]
                dv = self.dtab[t].val_tab[i+1]-v
                dm = self.dtab[t].mjd_tab[i+1]-self.dtab[t].mjd_tab[i]
                xm = mjd - self.dtab[t].mjd_tab[i]
                return v + dv * xm / dm
            else:
                return self.dtab[t].val_tab[i]

    def getrange(self, from_mjd, to_mjd):
        ft, fN = self.mjd2tabNoandindex(from_mjd)
        tt, tN = self.mjd2tabNoandindex(to_mjd)
        if fN is not None:
            fN = fN+1
        if (ft == tt and fN is None):
            return None
        out = MTSerie()
        if ft == -1:
            ft = 0
        if tt == -1:
            tt = 0

        for tab in self.dtab[ft:tt+1]:
            tmp = tab.getrange(from_mjd, to_mjd)
            if tmp is not None:
                out.add_TSerie(tmp)
        if len(out.dtab) == 0:
            return None
        return out

    def getrange_on_self(self, fmjd, tmjd):
        self.rmrange(0, fmjd)
        self.rmrange(tmjd, 1e6)

    def rmrange(self, fmjd, tmjd):
        ft, fN = self.mjd2tabNoandindex(fmjd)
        tt, tN = self.mjd2tabNoandindex(tmjd)
        if fN is not None:
            fN = fN+1
        if (ft == tt and fN is None):
            return self
        if ft == -1:
            ft = 0
        if tt == -1:
            tt = 0
        i = tt
        while i >= ft:
            x = self.dtab[i]
            (a, b) = x.rmrange(fmjd, tmjd)
            if b == 1:
                del self.dtab[i]
            if b == 2 or b == 3:
                self.dtab[i] = a
            if b == 4:
                del self.dtab[i]
                self.dtab.insert(i, a[1])
                self.dtab.insert(i, a[0])
            i = i-1

    def rm_indexes(self, indexes):
        for i, tab in enumerate(self.dtab):
            tab.rm_indexes(indexes[i])

    def rmoutlayers(self, max_iterations=4, target=None, maxdiff=None):
        i = 0
        no_rm = 0
        indexes_in_while_to_delete = []
        while (i < max_iterations and no_rm == 0):
            i += 1
            input_len = len(self.mjd_tab())
            if target is None:
                target = self.mean()
            if maxdiff is None:
                maxdiff = 3*self.std()
            indexes_in_mts_to_delete = []
            for x in self.dtab:
                indexes_in_mts_to_delete.append(
                    x.rmOutlayersOfTarget(
                        target=target,
                        maxDifference=maxdiff,
                    )
                )
            indexes_in_while_to_delete.append(indexes_in_mts_to_delete)
            if input_len == len(self.mjd_tab()):
                no_rm = 1
        return indexes_in_while_to_delete
    
    def rm_value(self, value, get_indexes_only=False):
        indexes_in_mts_to_delete = []
        for x in self.dtab:
            indexes_in_mts_to_delete.append(
                x.rm_value(value, get_indexes_only=get_indexes_only)
            )
        return indexes_in_mts_to_delete

    def mean(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        totlen = 0
        for x in self.dtab:
            if len(x.mjd_tab)>0:
                tab.append([x.mean, x.len])
                totlen = totlen+x.len
        out = 0
        for x in tab:
            out = out + x[0]*x[1]/totlen
        return out

    def mean_use_pps(self, decimal=False, decimal_out=False):
        if len(self.dtab) == 0:
            return None
        tab = []
        total_points=0
        for x in self.dtab:
            if len(x.mjd_tab)>0:
                mean, ts_points = x.mean_use_pps(
                    decimal=decimal,
                    decimal_out=decimal_out
                )
                tab.append((mean, ts_points))
                total_points=total_points+ts_points
        if decimal:
            out = D('0')
            for x in tab:
                out = out+D(x[0])*D(x[1])/D(total_points)
        else:
            out = 0
            for x in tab:
                out = out + x[0]*x[1]/total_points
        if decimal_out:
            return D(out)
        else:
            return float(out)

    def std(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        for x in self.dtab:
            tab = np.concatenate((tab, x.val_tab), axis=0)
        return np.std(tab)

    def stdm(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        totlen = 0
        for x in self.dtab:
            totlen = totlen + x.len
            tab = np.concatenate((tab, x.val_tab), axis=0)
        return np.std(tab)/np.sqrt(totlen)

    def len_mjd_eff(self):
        length = 0
        for x in self.dtab:
            length = length+x.len_mjd

    def mjd_tab(self):
        if len(self.dtab) > 0:
            tmp = [
                np.array(x.mjd_tab) for x in self.dtab
                if np.array(x.mjd_tab).ndim > 0
            ]
            if len(tmp) > 0:
                return np.concatenate(tmp)
        return []

    def val_tab(self):
        if len(self.dtab) > 0:
            tmp = [
                np.array(x.val_tab) for x in self.dtab
                if np.array(x.val_tab).ndim > 0
            ]
            if len(tmp) > 0:
                return np.concatenate(tmp)
        return []

    def saveToTxtFile(self, fileName):
        f = open(fileName, 'w')
        for x in self.dtab:
            i = 0
            length = len(x.mjd_tab)
            f.write('#\n')
            while i < length:
                f.write('%f\t%f\n' % (x.mjd_tab[i], x.val_tab[i]))
                i = i+1
        f.close()
    
    def first_mjd(self):
        return self.dtab[0].first_mjd()
    
    def last_mjd(self):
        return self.dtab[-1].last_mjd()
        
    def resample(self, fun='mean', period_s=60, start_mjd=None, points_ratio=0.7):
        self.rmemptyseries()
        first_mjd = self.first_mjd()
        first_mjd_int = np.floor(first_mjd)
        last_mjd = self.last_mjd()
        last_mjd_int = np.floor(last_mjd)
        period_mjd = period_s/(24*60*60)
        start_mjd = np.floor((first_mjd % 1)/period_mjd)*period_mjd + first_mjd_int
        stop_mjd = np.ceil((last_mjd % 1)/period_mjd)*period_mjd + last_mjd_int
        mjd_grid = np.arange(start_mjd, stop_mjd+1e-6, period_mjd)
        print(start_mjd, stop_mjd)
        return self.resample_to_mjd_array(
            mjd_grid=mjd_grid,
            grid_period_s=period_s,
            fun=fun,
            points_ratio=points_ratio,
        )

    def resample_to_mjd_array(
        self,
        mjd_grid,
        grid_period_s,
        fun='mean',
        points_ratio=0.7,
        none_fields=False,
        none_val=None,
    ):
        """
        params:
            mjd_grid: np.array 1D
        """

        sample_period_s = self.get_sample_period_s()
        expected_number_of_points = grid_period_s/sample_period_s
        period_mjd = grid_period_s/(24*60*60)
        ts=TSerie()
        for mjd in mjd_grid:
            sub_mts = self.getrange(mjd, mjd+period_mjd)
            if sub_mts and sub_mts.get_number_of_points() > expected_number_of_points*points_ratio:
                if fun=='mean':
                    calc = sub_mts.mean()
                if fun=='slope':
                    calc = sub_mts.slope()
                if fun=='slope_s':
                    calc = sub_mts.slope_s()
                if calc:
                    ts.append(
                        mjd=mjd,
                        val=calc,
                        pps=sub_mts.get_number_of_points()
                    )
                    ts.__str__()
                else:
                    ts.append(
                        mjd=mjd,
                        val=none_val,
                        pps=sub_mts.get_number_of_points()
                    )
            else:
                if none_fields:
                    ts.append(
                        mjd=mjd,
                        val=none_val,
                    )
                    ts.__str__()
        out_mts = MTSerie(TSerie=ts)
        out_mts.split(min_gap_s=grid_period_s*1.7)
        # out_mts.rmemptyseries()
        return out_mts

    def resample_to_mts_grid(
        self,
        mts,
        grid_period_s,
        fun='mean',
        points_ratio=0.7,
        none_fields=False,
        none_val=None,
    ):
        mjd_array = mts.mjd_tab()
        return self.resample_to_mjd_array(
            mjd_grid = mjd_array,
            grid_period_s=grid_period_s,
            fun=fun,
            points_ratio=points_ratio,
            none_fields=none_fields,
            none_val=none_val,
        )

    def get_sample_period_s(self):
        time = self.getTotalTimeWithoutGaps()
        points = self.get_number_of_points()
        return 24*60*60*time/points

    def get_number_of_points(self):
        num_of_points = 0
        for x in self.dtab:
            num_of_points += len(x.mjd_tab)
        return num_of_points
    
    def slope(self):
        number_of_points = self.get_number_of_points()
        if number_of_points < 2:
            return np.nan
        slope, intercept, r_value, p_value, str_err = linregress(
            self.mjd_tab(), self.val_tab()
        )
        return slope

    def slope_s(self):
        return self.slope()/(24*60*60)
    
    def time_diff_to_freq_diff(self):
        for x in self.dtab:
            x.time_diff_to_freq_diff()

    def add_val_offset_from_mts(self, mts):
        from_mjd = mts.dtab[0].mjd_tab[0]
        to_mjd = mts.dtab[-1].mjd_tab[-1]
        first_tab, first_index = self.mjd2tabNoandindex(from_mjd)
        last_tab, last_index = self.mjd2tabNoandindex(to_mjd)
        if last_index is None:
            last_index = len(self.dtab[last_tab].mjd_tab)-1
        if first_index is not None:
            first_index = first_index+1
        if (first_tab == last_tab and first_index is None):
            return self
        it = first_tab
        ii = first_index
        while (it<=last_tab or ii<=last_index):
            logging.info(
                f"{self.dtab[it].val_tab[ii]}, {mts.mjd2val(self.dtab[it].mjd_tab[ii])}"
            )
            if mts.mjd2val(self.dtab[it].mjd_tab[ii]) is not None:
                self.dtab[it].val_tab[ii]+=mts.mjd2val(self.dtab[it].mjd_tab[ii])
            if ii==len(self.dtab[it].mjd_tab)-1:
                ii=0
                it+=1
            else:
                ii+=1
    
    def add_sin(self, amplitude=0, omega=0):
        """
        Add amplitude*sin(omega*t) to existing data
        """

        for ts in self.dtab:
            ts.add_sin(amplitude=amplitude, omega=omega)

class TimePeriod:
    """
        Class for managing single continuous time period.
        Includes information only about start and stop mjd.
    """
    
    def __init__(self, start, stop):
        self.start = min(start, stop)
        self.stop = max(start, stop)

    def __str__(self):
        return str(self.start) + " -> " + str(self.stop)

    def __mul__(self, b):
        """
            Intersection of TimePeriods
        """
        
        if isinstance(b, TimePeriod):
            start = max(self.start, b.start)
            stop = min(self.stop, b.stop)
            if start < stop:
                return TimePeriod(start, stop)
        return None

    def joinIfOverlap(self, b):
        if isinstance(b, TimePeriod):
            if self*b is not None:
                start = min(self.start, b.start)
                stop = max(self.stop, b.stop)
                return TimePeriod(start, stop)
        return None

    def len(self):
        return self.stop-self.start


class TimePeriods:
    """
    Class for managing multiple time periods

    :Changes:
        2023-09-21 by Piotr Morzyński: Firs version
    """
    def __init__(self, periods=None):
        """
        :param periods - list of elements of type TimePeriod
        """
        self.periods = []
        if periods is not None:
            for period in periods:
                self.periods.append(period)

    def __str__(self):
        return "".join(['| '+x.__str__()+' ' for x in self.periods])

    def appendPeriod(self, periodToAdd):
        outputPeriods = []
        insertIndex = None
        for c, x in enumerate(self.periods):
            if x*periodToAdd is None:
                outputPeriods.append(x)
                if (insertIndex is None and
                        periodToAdd.stop < x.start):
                    insertIndex = c
            else:
                if insertIndex is None:
                    insertIndex = c
                periodToAdd = periodToAdd.joinIfOverlap(x)
        if insertIndex is None:
            insertIndex = len(outputPeriods)
        outputPeriods.insert(insertIndex, periodToAdd)
        self.periods = outputPeriods

    def appendPeriods(self, TimePeriodsToAdd):
        for x in TimePeriodsToAdd.periods:
            self.appendPeriod(x)

    def append(self, b):
        if isinstance(b, TimePeriod):
            self.appendPeriod(b)
        if isinstance(b, TimePeriods):
            self.appendPeriods(b)

    def commonPart(self, b):
        if isinstance(b, TimePeriod):
            outPeriods = []
            for x in self.periods:
                xb = x*b
                if xb is not None:
                    outPeriods.append(xb)
            if len(outPeriods) == 0:
                return None
            return TimePeriods(periods=outPeriods)
        if isinstance(b, TimePeriods):
            outPeriods = []
            for x in b.periods:
                cp = self.commonPart(x)
                if cp is not None:
                    outPeriods.append(cp)
            if len(outPeriods) == 0:
                return None
            return TimePeriods(periods=outPeriods)
        return None

    def totalTimeWithoutGaps(self):
        acc = 0
        for x in self.periods:
            acc = acc+x.len()
        return acc
    

class MGserie:
    def __init__(self, ser1, ser2, grid_s, fmjd=None, tmjd=None, grid_mode=1):
        self.dtab = []
        self.grid_s = grid_s
        self.grid_mjd = grid_s/(24*60*60)
        self.mjd_min = min(ser1.mjd_tab()[0], ser2.mjd_tab()[0])
        self.mjd_max = max(ser1.mjd_tab()[-1], ser2.mjd_tab()[-1])
        if fmjd is not None:
            self.mjd_min = max(self.mjd_min, fmjd)
        if tmjd is not None:
            self.mjd_max = min(self.mjd_max, tmjd)

        grid_tab = np.arange(self.mjd_min, self.mjd_max, self.grid_mjd)
        self.dtab = np.zeros((3, len(grid_tab)))
        for ind, mjd in enumerate(grid_tab):
            self.dtab[1][ind] = ser1.mjd2val(mjd, mode=grid_mode)
            self.dtab[2][ind] = ser2.mjd2val(mjd, mode=grid_mode)
        self.dtab[0] = grid_tab

    def corr(self, fmjd, tmjd, shift_s=None, norm=True):
        fi1 = self.mjd2index(fmjd)
        ti1 = self.mjd2index(tmjd)
        if shift_s is None:
            fi2 = fi1
            ti2 = ti1
        else:
            try:
                fi2 = fi1 + int(shift_s/self.grid_s)
                ti2 = ti1 + int(shift_s/self.grid_s)
            except:
                return np.nan
            if fi2 < 0 or ti2 > len(self.dtab[0])-1:
                return np.nan

        d1 = self.dtab[1][fi1:ti1]
        d2 = self.dtab[2][fi2:ti2]
        if len(d1) == len(d2):
            if norm is True:
                return np.correlate(d1, d2, 'valid')[0]/len(d1)
            else:
                return np.correlate(d1, d2, 'valid')[0]
        else:
            return np.nan

    def plot(self, ax=None, show=True, color='g'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        for x in [1, 2]:
            d = np.array([
                [self.dtab[0][i], self.dtab[x][i]] for
                i in range(0, len(self.dtab[0]))
                # if not np.isnan(self.dtab[x][i])
            ]).transpose()
            ax.plot(d[0], d[1], color=color)
        if show is True:
            plt.show()
        return ax

    def mjd2index(self, mjd, init_index=None):
        leni = len(self.dtab[0])
        len_mjd = self.dtab[0][-1]-self.dtab[0][0]
        if init_index is None:
            N = int((leni/len_mjd)*(mjd-self.dtab[0][0]))
        else:
            N = init_index
        if (mjd < self.dtab[0][0] or mjd > self.dtab[0][-1]):
            return None
        while 1:
            if self.dtab[0][N] > mjd:
                N = N-1
            else:
                if self.dtab[0][N+1] > mjd:
                    return N
                else:
                    N = N+1


class GTserie:
    """
    Class storing multiple time series of type MTserie
    """
    
    def __init__(self, name):
        self.name = name
        self.mts_dict = dict()
        self.number_of_mts = 0
        self.mjd_groups = dict()
    
    def __str__(self):
        return (
            f"{self.name}:\n"
            f"\tnumber of series: {self.number_of_mts}"
        )

    def append_mtserie(self, mts_name, mts, mjd_group=''):
        """
        Append MTSerie to GTserie

        Params:
            mts_name (str): name of MTSerie
            mts (MTSerie): data
            time_group (str): name of mjd group
        """
        self.mts_dict[mts_name] = mts
        self.number_of_mts = len(self.mts_dict)
        self.mjd_groups[mts_name] = mjd_group
    
    def import_data_rocit_oc(
        self,
        info,
        name='umk',
        headers = ['date', 'time', 'frac_freq', 'confidence', 'systematics'],
    ):
        df = import_data_to_df_rocit_oc(info=info, name=name, headers=headers)
        self.append_df_as_mtseries(df, name, columns=['frac_freq', 'confidence', 'systematics'])

    def import_data_rocit_gnss(self, path='./Data_storage/sn112-nmij.dat', name='gnss'):
        df = import_data_to_df_rocit_gnss(path=path)
        self.append_df_as_mtseries(df, name, ['delta_t_ns'])
        
    def append_df_as_mtseries(self, df, name, columns, mjd_name='mjd'):
        for column in columns:
            self.append_mtserie(
                mts_name=name+'_'+column,
                mts=MTSerie(
                    label=name+'_'+column,
                    TSerie=TSerie(mjd=df[mjd_name].to_numpy(), val=df[column].to_numpy())
                ),
                mjd_group=name+'_mjd',
            )
    
    def print_all_mts(self):
        for a in self.mts_dict:
            print(self.mts_dict[a])
    
    def plot_mts_one_by_one(self):
        for a in self.mts_dict:
            self.mts_dict[a].plot()
    
    def plot_mts(self, mts_name):
        self.mts_dict[mts_name].plot()

    def plot(self, fig=None, axs=None, figsize=(7, 7), mts_names=None, show=1, zorder=1):
        if not mts_names:
            mts_names = self.mts_dict
        fig, axs = plt.subplots(len(mts_names),1,  constrained_layout=True, sharex=True, figsize=figsize)
        for i, mts_name in enumerate(mts_names):
            self.mts_dict[mts_name].plot(ax=axs[i], show=0, zorder=zorder)
            axs[i].grid(True)
            axs[i].set_ylabel(self.mts_dict[mts_name].plot_label)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axs

    def get_mtss_from_mjd_group(self, mjd_group, exclude=None):
        keys = [key for key, value in self.mjd_groups.items() if value == mjd_group]
        if exclude in keys:
            keys.remove(exclude)
        return keys

    def rm_indexes_from_mjd_group(self, indexes, mjd_group, exclude):
        mtss = self.get_mtss_from_mjd_group(mjd_group=mjd_group, exclude=exclude)
        for mts in mtss:
            self.mts_dict[mts].rm_indexes(indexes)

    def rm_outlayers(self, mts_name):
        mts = self.mts_dict[mts_name]
        indexes_to_delete_iterated = mts.rmoutlayers()
        mjd_group = self.mjd_groups[mts_name]
        for indexes_to_delete in indexes_to_delete_iterated:
            self.rm_indexes_from_mjd_group(
                indexes=indexes_to_delete,
                mjd_group=mjd_group,
                exclude=mts_name
            )
    
    def rm_value(self, mts_name, value):
        mts = self.mts_dict[mts_name]
        indexes_in_mts_to_delete = mts.rm_value(value, get_indexes_only=True)
        mjd_group = self.mjd_groups[mts_name]
        self.rm_indexes_from_mjd_group(
            indexes=indexes_in_mts_to_delete,
            mjd_group=mjd_group,
            exclude=None
        )

    def rm_range(self, from_mjd, to_mjd):
        for a in self.mts_dict:
            self.mts_dict[a].rmrange(from_mjd, to_mjd)
    
    def split_mjd_group(self, mjd_group, min_gap_s=160):
        mtss = self.get_mtss_from_mjd_group(mjd_group)
        for mts in mtss:
            self.mts_dict[mts].__str__()
            self.mts_dict[mts].split(min_gap_s=min_gap_s)
    
    def split(self, min_gap_s):
        for mjd_group in self.mjd_groups:
            self.split_mjd_group(mjd_group=mjd_group, min_gap_s=min_gap_s)

    def resample(self, fun='mean', period_s=60, start_mjd=None, points_ratio=0.1):
        for a in self.mts_dict:
            print('************', a)
            self.mts_dict[a] = self.mts_dict[a].resample(
                fun=fun,
                period_s=period_s,
                start_mjd=start_mjd,
                points_ratio=points_ratio,
            )

    def resample_to_mts(
        self,
        ref_mts_name,
        grid_period_s,
        none_fields=True,
        rm_none_fields=True,
        none_val=None,
        new_gts=True,
        points_ratio=0.7,
    ):
        ref_mts = self.mts_dict[ref_mts_name]
        mjd_group = self.mjd_groups[ref_mts_name]
        # mtss_names = [mts_name for mts_name in self.mts_dict]
        if new_gts:
            g = GTserie(name='resampled')
        for mts_name in self.mts_dict:
            mts = self.mts_dict[mts_name]
            if self.mjd_groups[mts_name] == mjd_group:
                tmp = mts
            else:
                tmp = mts.resample_to_mts_grid(
                    mts=ref_mts,
                    grid_period_s=grid_period_s,
                    fun='mean',
                    points_ratio=points_ratio,
                    none_fields=none_fields,
                    none_val=none_val,
                )
            if not new_gts:
                self.mts_dict[mts_name]=tmp
                self.mts_dict[mts_name].__str__()
            else:
                g.append_mtserie(
                    mts_name=mts_name,
                    mts=tmp,
                    mjd_group=mjd_group,
                )
        if rm_none_fields:
            # for mts_name in g.mts_dict:
            #     g.rm_value(mts_name, none_val)
            print(
                two_mts_equal_mjd(
                    self.mts_dict['umk_frac_freq'],
                    self.mts_dict['nmij_frac_freq']
                )
            )
            # g.rm_value('nmij_frac_freq', none_val)
        if new_gts:
            return g     
        
    def add_mts_to_mts(self, mts_name_1, mts_name_2, mts_name_out):
        mts1=self.mts_dict[mts_name_1]
        mts2=self.mts_dict[mts_name_2]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            ts2 = mts2.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                val_tab.append(ts1.val_tab[i_ts]+ts2.val_tab[i_ts])
                mjd_tab.append(ts1.mjd_tab[i_ts])
            out_ts = TSerie(mjd=mjd_tab, val=val_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')

    def math_mts_and_mts(self, operation, mts_name_1, mts_name_2, mts_name_out, decimal=False):
        mts1=self.mts_dict[mts_name_1]
        mts2=self.mts_dict[mts_name_2]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            ts2 = mts2.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            pps_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                if operation in OPERATIONS:
                    val_tab.append(OPERATIONS[operation](
                        ts1.val_tab[i_ts],
                        ts2.val_tab[i_ts]
                    ))
                mjd_tab.append(ts1.mjd_tab[i_ts])
                pps_tab.append((ts1.pps_tab[i_ts]+ts1.pps_tab[i_ts])/2)
            out_ts = TSerie(mjd=mjd_tab, val=val_tab, pps=pps_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')
    
    def math_mts_and_number(self, operation, mts_name_1, number, mts_name_out):
        mts1=self.mts_dict[mts_name_1]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            pps_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                if operation in OPERATIONS:
                    val_tab.append(OPERATIONS[operation](
                        ts1.val_tab[i_ts],
                        number
                    ))
                mjd_tab.append(ts1.mjd_tab[i_ts])
                pps_tab.append(ts1.pps_tab[i_ts])
            out_ts = TSerie(mjd=mjd_tab, val=val_tab, pps=pps_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')


def import_data_to_df_rocit_oc(
    info,
    name='umk',
    headers = ['date', 'time', 'frac_freq', 'confidence', 'systematics'],
):
    """
    Importing data to dataframe (pandas) from March 2020 Rocit campaign and creating mjd data

    Args:
        info (dict): dictionary including information about path to lab/clock data
            ex.:
                data_path = path.Path('./Data_storage/clocks_vs_maser')
                info['umk'] ={'data_dir': data_path / 'UMK_Sr1-HMAOS'}
        name: name of the lab/clock
        headers: names of columns imported from file
    Return:
        Pandas dataframe
    """
    
    df = pd.DataFrame([])
    for f in info[name]['data_dir'].iterdir():
        if f.suffix == '.dat':
            p = pd.read_csv(
                f,
                names=headers,
                skiprows=11, 
                skip_blank_lines=True, 
                sep=' |\t',
                engine='python'
            )
            df = pd.concat([df, p], ignore_index=True)
    df['mjd']=Time(pd.to_datetime(df['date']+' '+df['time'])).mjd
    return df


def import_data_to_df_rocit_gnss(path='./Data_storage/sn112-nmij.dat'):
    headers = ['mjd', 'delta_t_ns']
    p = pd.read_csv(
        path, 
        names=headers,
        skip_blank_lines=True,
        sep='\t',
    )
    df = pd.DataFrame(p)
    return df


def two_mts_equal_mjd(mts1, mts2):
    return (
        len(mts1.dtab) == len(mts2.dtab)
        and all(np.array_equal(a, b) for a, b in zip(mts1.mjd_tab(), mts2.mjd_tab()))
    )

OPERATIONS = {
    'add': lambda x, y: x + y,
    'add_d': lambda x, y: float(D(x)+D(y)),
    'multiply': lambda x, y: x * y,
    'multiply_d': lambda x, y: float(D(x)*D(y)),
    'divide': lambda x, y: x / y,
    'divide_d': lambda x, y: float(D(x)/D(y)),
}


def get_test_tserie(fmjd = 50000, tmjd=50001, period_s=1, noise_ampl=1, mean_val=0):
    mjd_tab = np.arange(fmjd, tmjd, 1/(24*60*60))
    val_tab = np.random.uniform(mean_val-noise_ampl, mean_val+noise_ampl, size=mjd_tab.shape)
    return TSerie(mjd=mjd_tab, val=val_tab)
