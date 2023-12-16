import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import allantools as al
from astropy.convolution import Gaussian1DKernel, convolve
from .mtserie import MTSerie


class TSerie:
    def __init__(self, label='', mjd=[], val=[]):
        self.label = label
        self.mjd_tab = np.array(mjd)
        self.val_tab = np.array(val)
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
                    s = s+'\t%.6f\t%f\n' % (self.mjd_tab[i], self.val_tab[i])
            else:
                for i in range(0, 5):
                    s = s+'\t%.6f\t%f\n' % (self.mjd_tab[i], self.val_tab[i])
                s = s+'\t...\n'
                for i in range(self.len-5, self.len):
                    s = s+'\t%.6f\t%f\n' % (self.mjd_tab[i], self.val_tab[i])
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

    def mean(self):
        if len(self.val_tab) > 0:
            return np.mean(self.val_tab)
        else:
            return None

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
        # print(tab_i)
        for j in range(0, len(tab_i)-1):
            out_tab.append(TSerie(
                mjd=self.mjd_tab[tab_i[j]:tab_i[j+1]],
                val=self.val_tab[tab_i[j]:tab_i[j+1]]
            ))
        return out_tab

    def append(self, mjd, val):
        self.mjd_tab = np.append(self.mjd_tab, mjd)
        self.val_tab = np.append(self.val_tab, val)

    def last(self):
        out = TSerie()
        out.append(self.mjd_tab[-1], self.val_tab[-1])
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
                val=self.val_tab[(tN+1):]),
                2
            )
        if (fmjd > self.mjd_start and tmjd >= self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            return (TSerie(mjd=self.mjd_tab[:fN],
                           val=self.val_tab[:fN]), 3)
        if (fmjd > self.mjd_start and tmjd < self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            left = TSerie(mjd=self.mjd_tab[:fN], val=self.val_tab[:fN])
            right = TSerie(
                mjd=self.mjd_tab[(tN+1):],
                val=self.val_tab[(tN+1):]
            )
            return ([left, right], 4)

    def rm_outlayers_singledelta(self, max_delta):
        self.calc_tab()
        for i in range(1, self.len):
            if abs(self.val_tab[i] - self.val_tab[i-1]) > max_delta:
                print('outlayer detected')
                self.val_tab[i] = self.val_tab[i-1]

    def rmOutlayersOfTarget(self, target, maxDifference):
        i = 0
        # print('self.len: ', self.len)
        while i < self.len:
            if abs(self.val_tab[i]-target) > maxDifference:
                self.val_tab = np.delete(self.val_tab, i)
                self.mjd_tab = np.delete(self.mjd_tab, i)
                i = i-1
                self.len -= 1
            i = i+1

    def time_shift(self, sec):
        self.mjd_tab = self.mjd_tab+sec/(24*60*60)
        self.calc_tab()

    def show(self):
        N = 5
        if len(self.mjd_tab) > 2*N:
            for x in range(0, 5):
                print(self.mjd_tab[x], self.val_tab[x])
            print('...')
            for x in range(self.length-6, self.length-1):
                print(self.mjd_tab[x], self.val_tab[x])
        elif len(self.mjd_tab) > 0:
            for x in range(0, 2*N-1):
                print(self.mjd_tab[x], self.val_tab[x])
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

    def plot_allan(self, atom='88Sr'):
        if atom == '88Sr':
            fabs = 429228066418012
        t = np.power(10, np.arange(1, int(np.log10(self.len_s))+0.1, 0.1))
        y = self.val_tab/fabs
        r = self.len_s/self.len
        a = al.Dataset(data=y, rate=r, data_type="freq", taus=t)
        a.compute('adev')
        b = al.Plot()
        b.plot(a, errorbars=True, grid=True)
        b.show()

    def scatter(self):
        plt.scatter(self.mjd_tab, self.val_tab)
        plt.show()

    def rm_first(self, n):
        self.val_tab = np.delete(self.val_tab, np.s_[0:n], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[0:n], None)
        self.calc_tab()

    def rm_last(self, n):
        self.val_tab = np.delete(self.val_tab, np.s_[-n:], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[-n:], None)
        self.calc_tab()

    def gauss_filter(self, stddev=50):
        g = Gaussian1DKernel(stddev=stddev)
        self.val_tab = convolve(self.val_tab, g)
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)

    def high_gauss_filter(self, stddev=50):
        g = Gaussian1DKernel(stddev=stddev)
        tmp = convolve(self.val_tab, g)
        self.val_tab = self.val_tab - tmp
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)

    def toMTSerie(self):
        return MTSerie(TSerie=self)
