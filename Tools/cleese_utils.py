import os
import time
import shutil
import re
import numpy as np
import soundfile as sf
from cleese.cleese import cleeseProcess


def list_spkr_dirnames(dir_of_spkrs):
    spkr_dir_list = []
    for spkr_id in os.listdir(dir_of_spkrs):
        # print(spkr_id)
        if os.path.isdir(os.path.join(dir_of_spkrs, spkr_id)):
            spkr_dir_list.append(os.path.join(dir_of_spkrs, spkr_id))
    
    return spkr_dir_list
        

def list_file_paths (dirname):
    filepathlist = []
    
    for root, directories, files in os.walk(dirname):
        for filename in (files):
            filepath = os.path.join(root, filename)
            filepathlist.append(filepath)
        return filepathlist


def wavRead(fileName):

    #sr,wave = wav.read(fileName)
    wave,sr = sf.read(fileName)

    sampleFormat = wave.dtype

    if sampleFormat in ('int16','int32'):
        # convert to float
        if sampleFormat == 'int16':
            n_bits = 16
        elif sampleFormat == 'int32':
            n_bits = 32
        wave = wave/(float(2**(n_bits - 1)))
        wave = wave.astype('float32')

    return wave,sr,sampleFormat


def wavWrite(waveOut, fileName, sr, sampleFormat='int16'):

    if sampleFormat == 'int16':
        waveOutFormat = waveOut * 2**15
    elif sampleFormat == 'int32':
        waveOutFormat = waveOut * 2**31
    else:
        waveOutFormat = waveOut
    waveOutFormat = waveOutFormat.astype(sampleFormat)
    #wav.write(fileName, sr, waveOutFormat)
    sf.write(fileName, waveOutFormat, sr)


def process(soundData, configFile, outputDirPath, transfer, BPF=None, sr=None, timeVec=None):
    data = {}
    exec(open(configFile).read(),data)
    pars = data['pars']

    doCreateBPF = False
    if BPF is None:
        doCreateBPF = True

    try:
        basestring
    except NameError:
        basestring = str

    if isinstance(soundData, basestring):
        fileInput = True
        waveIn,sr,sampleFormat = wavRead(soundData)     # read input sound file
    else:
        fileInput = False
        waveIn = soundData
        numFiles = 1

    if len(waveIn.shape)==2:
        print('Warning: stereo file detected. Reading only left channel.')
        waveIn = np.ravel(waveIn[:,0])

    pars['main_pars']['inSamples'] = len(waveIn)
    pars['main_pars']['sr'] = sr

    if doCreateBPF and fileInput:

        numFiles = pars['main_pars']['numFiles']

        # generate experiment name and folder
        if pars['main_pars']['generateExpFolder']:
            pars['main_pars']['expBaseDir'] = os.path.join(outputDirPath,time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            pars['main_pars']['expBaseDir'] = outputDirPath
        if not os.path.exists(pars['main_pars']['expBaseDir']):
            os.makedirs(pars['main_pars']['expBaseDir'])

        # copy base audio to experiment folder
        shutil.copy2(soundData,pars['main_pars']['expBaseDir'])

        # copy configuration file to experiment folder
        shutil.copy2(configFile,pars['main_pars']['expBaseDir'])

    else:

        numFiles = 1
        if np.isscalar(BPF):
            BPF = np.array([[0.,float(BPF)]])

    num_transf = len(pars['main_pars']['transf'])
    # when passing a given BPF, allow only one transformation
    if not doCreateBPF and len(pars['main_pars']['transf']) > 1:
        print("Warning: when passing a given BPF, only one transfomation is allowed. Applying the first transformation in the chain...")
        num_transf = 1

    currOutFile = []
    for t in range(0,num_transf):

        tr = transfer #pars['main_pars']['transf'][t]

        if pars['main_pars']['chain']:
            if t==0:
                currTrString = tr
            else:
                currTrString = currTrString+'_'+tr
        else:
            currTrString = tr

        if doCreateBPF:

            if fileInput:
                pars['main_pars']['currOutPath'] = os.path.join(pars['main_pars']['expBaseDir'],currTrString)

            # create BPF time vector
            duration = pars['main_pars']['inSamples']/float(sr)
            BPFtime, numPoints, endOnTrans = cleeseProcess.createBPFtimeVec(duration,pars[tr+'_pars'],timeVec=timeVec)

        else:
            pars['main_pars']['currOutPath'] = outputDirPath

        # create output folder
        if fileInput:
            if not os.path.exists(pars['main_pars']['currOutPath']):
                os.makedirs(pars['main_pars']['currOutPath'])
            path,inFileNoExt = os.path.split(soundData)
            inFileNoExt = os.path.splitext(inFileNoExt)[0]

        # create frequency bands for random EQ
        eqFreqVec = None
        if tr == 'eq':
            eqFreqVec = cleeseProcess.createBPFfreqs(pars)

        for i in range(0,numFiles):

            # print(currTrString+' variation '+str(i+1)+'/'+str(numFiles))
            currFileNo = "%04u" % (i+1)

            if pars['main_pars']['chain'] and t>0:
                if fileInput:
                    waveIn,sr,sampleFormat = cleeseProcess.wavRead(currOutFile[i])
                else:
                    waveIn = waveOut
                pars['main_pars']['inSamples'] = len(waveIn)
                pars['main_pars']['sr'] = sr
                BPFtime, numPoints, endOnTrans = cleeseProcess.createBPFtimeVec(duration,pars[tr+'_pars'],timeVec=timeVec)

            # generate random BPF
            if doCreateBPF:
                BPF = cleeseProcess.createBPF(tr,pars,BPFtime,numPoints,endOnTrans,eqFreqVec)

            # export BPF as text file
            if fileInput:
                # find the pitch factor by matching all numeric chars after the string 'pitch' by using a positive lookbehind
                matches = re.findall(r"(?<=pitch)[0-9]+", pars['main_pars']['currOutPath'].split('/')[-2])
                # if the current augmentation is 'pitch', add the pitch factor to the output txt and wav filenames being created
                pitch_lvl = str(matches[0]) if currTrString == 'pitch' else ''

                # manually set if you want to use the original cleese.process filenames structure
                long_filenames = False

                if long_filenames:
                    currBPFfile = long_BPFfilename(pars, currTrString, inFileNoExt, currFileNo, pitch_lvl)
                else:
                    currBPFfile, currTrString = short_BPFfilename(pars, currTrString, inFileNoExt, pitch_lvl)
                np.savetxt(currBPFfile,BPF,'%.8f')

                if t==0:
                    if long_filenames:
                        currOutFile.append(os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+'.'+currFileNo+'.'+currTrString+pitch_lvl+'.wav'))
                    else:
                        currOutFile.append(os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+currTrString+pitch_lvl+'.wav'))
                else:
                    if long_filenames:
                        currOutFile[i] = os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+'.'+currFileNo+'.'+currTrString+pitch_lvl+'.wav')
                    else:
                        currOutFile[i] = os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+currTrString+pitch_lvl+'.wav')

            if tr in ['stretch','pitch']:

                if tr == 'stretch':
                    doPitchShift = False
                elif tr == 'pitch':
                    doPitchShift = True

                # call processing with phase vocoder
                waveOut = cleeseProcess.processWithPV(waveIn=waveIn, pars=pars, BPF=BPF, doPitchShift=doPitchShift)

                # remove trailing zero-pad
                if tr == 'pitch':
                    waveOut = np.delete(waveOut, range(pars['main_pars']['inSamples'],len(waveOut)))

            elif tr == 'eq':

                waveOut = cleeseProcess.processWithSTFT(waveIn=waveIn, pars=pars, BPF=BPF)

                # remove trailing zero-pad
                waveOut = np.delete(waveOut, range(pars['main_pars']['inSamples'],len(waveOut)))

            elif tr == 'gain':

                if numPoints == 1:
                    waveOut = waveIn * BPF[:,1]
                else:
                    gainVec = np.interp(np.linspace(0,duration,pars['main_pars']['inSamples']),BPF[:,0],BPF[:,1])
                    waveOut = waveIn * gainVec

            if fileInput:
                # normalize
                if np.max(np.abs(waveOut)) >= 1.0:
                    waveOut = waveOut/np.max(np.abs(waveOut))*0.999
                wavWrite(waveOut,fileName=currOutFile[i],sr=sr,sampleFormat=sampleFormat)

    if not fileInput:
        return waveOut,BPF
    return currBPFfile.split('/')[-1].split('_')[-2]


def long_BPFfilename(pars, currTrString, inFileNoExt, currFileNo, pitch_lvl):
    # original structure of cleese.process filenames
    currBPFfile = os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+'.'+currFileNo+'.'+currTrString+pitch_lvl+'_BPF.txt')

    return currBPFfile


def short_BPFfilename(pars, currTrString, inFileNoExt, pitch_lvl):
    # creates shorter filenames, returns the modified currTrString

    # shortens 'pitch' to 'p' and for formatting purposes add a dot before p
    currTrString = '.p' if currTrString == 'pitch' else currTrString
    # do not add 'stretch' augmentation keyword to filename string
    currTrString = '' if currTrString == 'stretch' else currTrString
    # omit currFileNo from filename
    currBPFfile = os.path.join(pars['main_pars']['currOutPath'],inFileNoExt+currTrString+pitch_lvl+'_BPF.txt')

    return currBPFfile, currTrString