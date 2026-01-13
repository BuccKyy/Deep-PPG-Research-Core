import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import lfilter


class FindPeak:
    @staticmethod
    def avgfilter(signal, windows):
        for inx in range(len(signal)):
            if inx < windows // 2:
                signal[inx] = np.mean(signal[:inx + windows // 2])
            else:
                signal[inx] = np.mean(signal[inx - windows // 2:inx + windows // 2])
        return signal

    def findpeaks(self, data, spacing=1, limit=None):
        """Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param data: values
        :param spacing: minimum spacing to the next peak (should be 1 or more)
        :param limit: peaks should have value greater or equal
        :return:
        """
        ln = data.size
        x = np.zeros(ln + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + ln] = data
        peak_candidate = np.zeros(ln)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + ln]  # before
            start = spacing
            h_c = x[start: start + ln]  # central
            start = spacing + s + 1
            h_a = x[start: start + ln]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


class BpAnnotate:
    def __init__(self):
        self.Fs = 200

    def BP_resample(self, signal, orig_fs):
        if orig_fs != self.Fs:
            # BP_RESAMPLE Resample to 200 Hz
            duration = len(signal)/ orig_fs
            oldx = np.linspace(0, duration, len(signal))
            newx = np.linspace(0, duration, int(self.Fs*duration))

            resamp_sig = interpolate.PchipInterpolator(oldx, signal)(newx)
        else:
            duration = len(signal) / orig_fs
            oldx = np.linspace(0, duration, len(signal))
            newx = oldx
            resamp_sig = signal

        return resamp_sig, newx, oldx

    @staticmethod
    def BP_lowpass(signal):
        # LOWPASS Filter input signal (200 Hz assumed)
        b = np.array([1, 0, 0, 0, 0, 0, - 2, 0, 0, 0, 0, 0, 1])
        a = np.array([1, - 2, 1]) * 36
        unit = np.concatenate((np.array([1]), np.zeros(12)))

        h_l = lfilter(b, a, unit)
        filtwaveform = np.convolve(signal, h_l)
        filtwaveform = np.roll(filtwaveform, -5)
        filtwaveform[:5] = np.nan
        filtwaveform = filtwaveform[: len(signal)]
        return filtwaveform

    def doubleDerive(self, signal):
        waveformD = np.diff(signal)
        waveformD = np.concatenate((waveformD, np.ones(1)*np.nan))

        waveformDD = np.diff(waveformD)
        waveformDD = np.concatenate((waveformDD, np.ones(1) * np.nan))
        waveformDD = self.BP_lowpass(waveformDD)

        # Perform the switch for positive and negative first-derivative values
        waveformDDPlus = waveformDD * ((waveformD > 0) & (waveformDD > 0))
        waveformDDPlus = waveformDDPlus ** 2
        return waveformDDPlus, waveformDD, waveformD

    @staticmethod
    def rollingWindow(vector, winsize):
        vecsize = len(vector)
        rwin = np.ones((int(winsize), vecsize))*np.nan
        for i in range(int(winsize)):
            if i == 0:
                tmp = vector[:]
            else:
                tmp = vector[: -i]
            rwin[i, i:] = tmp
        return rwin

    @staticmethod
    def winsum(rwin):
        shape = np.shape(rwin)
        vecsize = shape[1]
        res = np.zeros(vecsize)
        for i in range(vecsize):
            # res[i] = sum(rwin[:, i])
            res[i] = np.nansum(rwin[:, i])
        return res

    @staticmethod
    def winmean(rwin, quant):
        shape = np.shape(rwin)
        vecsize = shape[1]
        res = np.zeros(vecsize)
        for i in range(vecsize):
            # res[i] = quant * np.mean(rwin[:, i])
            res[i] = quant * np.nanmean(rwin[:, i])
        return res

    @staticmethod
    def FixIndex(BrokeIndex, signal, Down, minWavelength):
        # follows a slope either up or down in a given window until an
        # extremum is attained
        BrokeIndex = np.asarray([BrokeIndex]).flatten()
        fixedIndex = BrokeIndex.copy()
        radius = round(minWavelength/4)
        for N in range(len(BrokeIndex)):
            # if N == len(BrokeIndex) - 1:
            #     a=0
            if BrokeIndex[N] + round(minWavelength) + 1 < len(signal):
                oldIndex = BrokeIndex[N]
                if Down:
                    if oldIndex + radius + 1 <= len(signal):
                        tempSignal = signal[int(np.max([oldIndex - radius, 1])): int(1 + oldIndex + radius)]
                    else:
                        tempSignal = signal[int(np.max([oldIndex - radius, 1])):]
                    newIndex = np.nanargmin(tempSignal)
                    newIndex = newIndex + oldIndex - radius - 1 + 1
                    while ~(oldIndex == newIndex):
                        oldIndex = newIndex
                        if oldIndex + radius + 1 <= len(signal):
                            tempSignal = signal[int(np.max([oldIndex - radius, 1])): int(1 + oldIndex + radius)]
                        else:
                            tempSignal = signal[int(np.max([oldIndex - radius, 1])):]
                        newIndex = np.nanargmin(tempSignal)
                        newIndex = newIndex + np.max([oldIndex - radius, 1]) - 1 + 1
                else:
                    if oldIndex + radius + 1 <= len(signal):
                        tempSignal = signal[int(np.max([oldIndex - radius, 1])): int(1 + oldIndex + radius)]
                    else:
                        tempSignal = signal[int(np.max([oldIndex - radius, 1])):]
                    newIndex = np.nanargmax(tempSignal)
                    newIndex = newIndex + oldIndex - radius - 1 + 1
                    while ~(oldIndex == newIndex):
                        oldIndex = newIndex
                        if oldIndex + radius + 1 <= len(signal):
                            tempSignal = signal[int(np.max([oldIndex - radius, 1])): int(1 + oldIndex + radius)]
                        else:
                            tempSignal = signal[int(np.max([oldIndex - radius, 1])):]
                        newIndex = np.nanargmax(tempSignal)
                        newIndex = newIndex + np.max([oldIndex - radius, 1]) - 1 + 1
                index = newIndex
                fixedIndex[N] = index
        fixedIndex = fixedIndex[fixedIndex < len(signal)]
        fixedIndex = fixedIndex[fixedIndex >= 0]
        return fixedIndex.astype(int)

    def getFootIndex(self, signal, waveformDDPlus, zoneOfInterest):
        zoneWall = np.diff(zoneOfInterest)
        BP_start = np.flatnonzero(zoneWall == 1)
        BP_stop = np.flatnonzero(zoneWall == -1)

        # Remove leading falling edges
        while BP_stop[0] < BP_start[0]:
            BP_stop = BP_stop[1:]
        nfeet = np.min([np.size(BP_start), np.size(BP_stop)])
        footIndex = np.zeros(nfeet)
        for i in range(nfeet):
            footIndex[i] = np.nanargmax(waveformDDPlus[BP_start[i]: BP_stop[i] + 1])
            footIndex[i] = footIndex[i] + BP_start[i] - 1 + 1
        zoneWall = np.diff(zoneOfInterest)
        BP_start = np.flatnonzero(zoneWall == 1)
        BP_stop = np.flatnonzero(zoneWall == -1)

        # Remove leading falling edges
        while BP_stop[0] < BP_start[0]:
            BP_stop = BP_stop[1:]

        nfeet = np.min([np.size(BP_start), np.size(BP_stop)])
        footIndex = np.zeros(nfeet)
        for i in range(nfeet):
            footIndex[i] = np.nanargmax(waveformDDPlus[BP_start[i]:  BP_stop[i] + 1])
            footIndex[i] = footIndex[i] + BP_start[i] - 1 + 1

        Down = 1
        # Depending on the morphology of the signal, the foot index may be
        # identified as the minimum of the waveform itself
        # { uncomment for local signal minimum identification
        footIndex = self.FixIndex(footIndex, signal, Down, 10)
        return footIndex

    def getDicroticIndex(self, waveformDD, waveformD, signal, footIndex, systolicIndex):
        if len(footIndex) == 1:
            print("Find only 1 foot peak in signal")
        RR = np.median(footIndex[1:] - footIndex[: - 1]) / self.Fs  # This assumes steady heartrate
        Down = 1
        Up = 1 - Down
        minWavelength = round(RR / 5 * self.Fs)

        straightLines = np.zeros(len(signal))
        notQuiteSystolicIndex = self.FixIndex(systolicIndex, signal, Up, minWavelength)

        for i in range(len(footIndex) - 1):
            # compute the parameters of the straight line going from systole to diastole
            slope = (signal[footIndex[i + 1]] - signal[notQuiteSystolicIndex[i]]) / (footIndex[i + 1] - notQuiteSystolicIndex[i])
            intercept = signal[footIndex[i + 1]] - slope * (footIndex[i + 1] + 1)
            straightLines[footIndex[i]: notQuiteSystolicIndex[i]+1] = signal[footIndex[i]: notQuiteSystolicIndex[i] + 1]
            straightLines[notQuiteSystolicIndex[i]: footIndex[i + 1]+1] = slope * (np.arange(notQuiteSystolicIndex[i], footIndex[i+1]+1)+1) + intercept

        eyeBallSignal = signal - straightLines
        eyeBallSignal[footIndex[-1]:] = waveformDD[footIndex[-1]:]

        notchIndex = self.FixIndex(systolicIndex + round(minWavelength), eyeBallSignal, Down, minWavelength / 4)
        dicroticIndex = self.FixIndex(notchIndex + round(0.25 * minWavelength), waveformDD, Down, round(0.25 * minWavelength))
        systolicIndex = systolicIndex[:len(dicroticIndex)]

        # if a local minimum and maximum exist, move the dicrotic indices to these
        for i in range(len(systolicIndex)):
            Start = int(systolicIndex[i] + round(minWavelength / 2))
            End = int(np.min([dicroticIndex[i] + round(minWavelength / 4), len(waveformD)]))
            ZOI = waveformD[Start: End + 1]
            ZOI = ZOI[1:] * ZOI[:- 1]
            extrema = np.flatnonzero(ZOI < 0)
            if len(extrema) >= 2:
                notchIndex[i] = self.FixIndex(notchIndex[i], signal, Down, 4)
                dicroticIndex[i] = self.FixIndex(np.min([notchIndex[i] + round(0.25 * minWavelength), len(signal)]), signal, Up, 4)
        return dicroticIndex, notchIndex

    def process(self, signal, orig_fs, unit, isClean=True, plot=False):
        integwinsize = np.floor(self.Fs / 4)
        # threswinsize = np.floor(self.Fs * 3)
        threswinsize = np.floor(self.Fs * 1)

        # resample the time-series to allow standardisation
        resamp_sig, time, origtime = self.BP_resample(signal, orig_fs)

        # filter
        filt_sig = self.BP_lowpass(resamp_sig)

        # derivatives
        waveformDDPlus, waveformDD, waveformD = self.doubleDerive(filt_sig)

        # Deal with very large data sets
        sizeLimit = 3 * 1e5

        if len(filt_sig) > sizeLimit:
            print('Data set exceeds {} size limit, performing sub-windowing.'.format(sizeLimit))
            BP_integral = np.zeros(len(filt_sig))
            threshold = np.zeros(len(filt_sig))

            numSubParts = np.ceil(len(filt_sig) / sizeLimit)
            overlap = round((numSubParts * sizeLimit - len(filt_sig)) / (numSubParts - 1))
            for i in range(1, numSubParts+1):
                Start = ((i - 1) * sizeLimit) - (i - 1) * overlap + 1
                End = min(Start + sizeLimit - 1, len(filt_sig))
                subIntegralWindow = self.rollingWindow(waveformDDPlus[Start: End], integwinsize)
                subIntegral = self.winsum(subIntegralWindow)
                subIntegral = np.roll(subIntegral, -np.floor(integwinsize / 2), axis=1)

                # implementation of Sun et al. paper A Signal Abnormality Index for Arterial Blood Pressure Waveform
                if unit == 'mmHg':
                    # negative slope for noise detection
                    subNoiseWindow = self.rollingWindow(waveformD[Start: End+1], np.floor(integwinsize / 2))
                    subNoiseLevel = self.winsum(subNoiseWindow)
                subThresholdWindow = self.rollingWindow(subIntegral, threswinsize)
                subThreshold = self.winmean(subThresholdWindow, 1.5)

                BP_integral[Start + overlap: End + 1] = subIntegral[1 + overlap:]
                threshold[Start + overlap: End + 1] = subThreshold[1 + overlap:]
                noiseLevel = np.ones(len(filt_sig))*np.nan
                if unit == 'mmHg':
                    noiseLevel[Start + overlap: End+1] = subNoiseLevel[1+overlap:]

                if i > 1:
                    BP_integral[Start: Start + overlap + 1] = np.mean(np.array([BP_integral[Start: Start + overlap + 1], subIntegral[: 2 + overlap]]))
                    threshold[Start: Start + overlap + 1] = np.mean(np.array([threshold[Start: Start + overlap + 1], subThreshold[: 2 + overlap]]))
                    if unit == 'mmHg':
                        noiseLevel[Start: Start + overlap + 1] = np.nanmean(np.array([noiseLevel[Start: Start + overlap + 1], subNoiseLevel[: 2 + overlap]]))
                else:
                    BP_integral[Start: Start + overlap + 1] = subIntegral[: 2 + overlap]
                    threshold[Start: Start + overlap + 1] = subThreshold[: 2 + overlap]
                    if unit == 'mmHg':
                        noiseLevel[Start: Start + overlap + 1] = subNoiseLevel[: 2 + overlap]
        else:
            # Moving sum to increase SNR
            integralWindow = self.rollingWindow(waveformDDPlus, integwinsize)
            BP_integral = self.winsum(integralWindow)

            # implementation of Sun et al. paper A Signal Abnormality Index for Arterial Blood Pressure Waveform
            if unit == 'mmHg':
                noiseWindow = self.rollingWindow(waveformD, np.floor(integwinsize / 2))
                noiseLevel = self.winsum(noiseWindow)

            # Center the integral
            BP_integral = np.roll(BP_integral, -int(np.floor(integwinsize / 2)), axis=0)

            thresholdWindow = self.rollingWindow(BP_integral, threswinsize)
            threshold = self.winmean(thresholdWindow, 1.5)

        # isClean forces the identification of indices before the window has had time to initiate
        if isClean:
            firstNotNan = np.flatnonzero(~np.isnan(threshold))
            firstNotNan = np.mean(threshold[firstNotNan[0]: firstNotNan[0] + int(integwinsize) + 1])
            threshold[np.isnan(threshold)] = firstNotNan * np.ones(len(threshold[np.isnan(threshold)]))

        # each zone of interest corresponds to a heart beat
        BP_integral[np.isnan(BP_integral)] = 0
        zoneOfInterest = (BP_integral > threshold)*1

        # implementation of Sun et al. paper A Signal Abnormality Index for Arterial Blood Pressure Waveform
        if unit == "mmHg":
            zoneOfInterest[filt_sig > 300] = 0
            zoneOfInterest[filt_sig < 20] = 0
            zoneOfInterest[noiseLevel < -40] = 0

        footIndex = self.getFootIndex(filt_sig, waveformDDPlus, zoneOfInterest)

        Down = 1
        Up = 1 - Down
        systolicIndex = self.FixIndex(footIndex + np.floor(integwinsize/2), filt_sig, Up, np.floor(integwinsize/2))

        # implementation of Sun et al. paper A Signal Abnormality Index for Arterial Blood Pressure Waveform
        if unit == 'mmHg':
            meanPressure = (filt_sig[systolicIndex] + filt_sig[footIndex]) / 2
            pulsePressure = filt_sig[systolicIndex] - filt_sig[footIndex]
            footIndex[(meanPressure < 30) | (meanPressure > 200) | (pulsePressure < 20)] = []
            systolicIndex[(meanPressure < 30) | (meanPressure > 200) | (pulsePressure < 20)] = []

        dicroticIndex, notchIndex = self.getDicroticIndex(waveformDD, waveformD, filt_sig, footIndex, systolicIndex)
        footIndex = footIndex[:len(notchIndex)]
        systolicIndex = systolicIndex[:len(notchIndex)]

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
            ax1.plot(time, filt_sig, label='Filtered')
            ax1.plot(origtime, signal, label='Waveform')
            ax1.plot(time[footIndex], filt_sig[footIndex], 'k*', label='Foot')
            ax1.plot(time[systolicIndex], filt_sig[systolicIndex], 'g*', label='Systole')
            ax1.plot(time[notchIndex], filt_sig[notchIndex], 'b*', label='Notch')
            ax1.plot(time[dicroticIndex], filt_sig[dicroticIndex], 'r*', label='Discrotic Peak')
            ax1.legend()
            ax1.set_ylabel('Arterial pressure')

            ax2.plot(time, waveformDD, label='2nd Derivative')
            ax2.plot(time, BP_integral, label='Integral')
            ax2.plot(time, threshold, label='Threshold')
            ax2.plot(time, zoneOfInterest*0.1, label='ZOI')
            ax2.legend()
            ax2.set_xlabel('Time (s)')

            plt.show()
        return footIndex, systolicIndex, notchIndex, dicroticIndex, np.nan_to_num(filt_sig), np.nan_to_num(waveformD), np.nan_to_num(waveformDD), waveformDDPlus
