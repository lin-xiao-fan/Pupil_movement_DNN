import csv, pandas, scipy
import numpy as np
import matplotlib.pyplot as plt

SUBJECTNUM='2025-01-24_1104'
AFE_FS=250

def afe_raw2val(data_raw):
    afe_data=[]

    # Convert first 4 bytes in each row into int32 timestamp
    # Group other bytes in 3 and convert then into int32 values
    for row in data_raw:
        row_val=[]
        # Timestamp conversion
        timestamp=np.frombuffer(row[0:4], dtype='u1').view(dtype='>i4')[0]
        row_val.append(timestamp)

        # Data conversion
        values_raw=np.frombuffer(row[4:], dtype='u1').reshape(-1,3)
        values=[]
        for group in values_raw:
            # Ensure the group has three bytes by padding with zeros on the left if necessary
            int32_value = np.pad(group, (1, 0), 'constant').view(dtype='>i4')[0]
            values.append((int32_value << 10) / (2**10))
        
        row_val.extend(values)

        afe_data.append(row_val)

    afe_data=np.asarray(afe_data,dtype='>i4')

    return afe_data

afe_data_raw=[]

afe_value=afe_raw2val(afe_data_raw)


#'''
afe_data=afe_value
afe_data_f=np.delete(afe_data,0, axis=1)
afe_data_f=afe_data_f.flatten()
# afe_data_phases=np.zeros([12,int(afe_data_f.size/12)])
afe_data_phases=np.reshape(afe_data_f,(-1,12)).T 

afe_data_phases=afe_data_phases / (2**22) * 1.2   #From ADC CODE to Full-scale voltage


afe_start_time=afe_data[0,0]

#'''

###

#afe_data= np.genfromtxt((f'./{SUBJECTNUM}/{SUBJECTNUM}_afe.csv'), delimiter=",", skip_header=1)

###
afe_time=np.arange(0,afe_data_phases.shape[1]/AFE_FS, 1/AFE_FS)


def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a=scipy.signal.butter(order, lowcut, fs=fs, btype='lowpass')
    return scipy.signal.filtfilt(b,a,data)

def butter_highpass_filter(data, highcut, fs, order=5):
    b, a=scipy.signal.butter(order, highcut, fs=fs, btype='highpass')
    return scipy.signal.filtfilt(b,a,data)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a=scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    return scipy.signal.filtfilt(b,a,data)

afe_data_filtered=[]

def butter_bandpass_filter1( lowcut, highcut, fs , order=5):
    b, a=scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    print( "b : ", b )
    print( "a : ", a )

butter_bandpass_filter1(0.01, 30, fs=250, order=2)
for row in afe_data_phases:
    afe_data_filtered.append(butter_bandpass_filter(row, 0.01, 30, fs=AFE_FS, order=2))


afe_data_filtered=np.asarray(afe_data_filtered)

print("濾波後的數據：")
print(afe_data_filtered)





















