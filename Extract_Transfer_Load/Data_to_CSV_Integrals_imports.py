import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def transform_data(csvfile):

  df = pd.read_csv(csvfile, header=None)

  _A_=df.values.tolist()[2][1:]
  _C_=df.values.tolist()[1][1:]
  _E_ = df.values.tolist()[0][1:]
  isSpincoated = df.values.tolist()[3][1:]
  isIncreasing = df.values.tolist()[4][1:]
  temperature_values = df.values.tolist()[5][1:]
  isRepeatUse = df.values.tolist()[6][1:]
  num_days = df.values.tolist()[7][1:]

  fcl_values = df.values.tolist()[8]
  fcl_values = [x for x in fcl_values if str(x) != 'nan' and 'omit' not in str(x).lower()]


  ppm = []

  for i in fcl_values:
    try:
      ppm.append(float(i))
    except:
      pass

  exp_params = df.values.tolist()
  params = []

  for i in range (0, 8):
      params.append(exp_params[i][0])

  df2 = df[9:].dropna(axis='columns')



  k=0
  times=[]
  currents=[]
  amps=[]


  for i,j in df2.iterrows():
    curr = list(map(float,j.values))

    times.append(curr[0])

    currents.append(curr[1:])
    
    for i in curr[1:]:
      amps.append(i)

  length=len(currents[18])

  
  x_vals = [eval(i) for i in df2[0].values.tolist()]
  y_integrals = []
  diffs = []

  longest_range_for_diff = 0
  
  # tmps = [df2[i].values.tolist() for i in range(1, length)]


  for i in range(1, length+1):
    tmp = df2[i].values.tolist()

    diff_tmp = np.gradient(tmp, x_vals)
    diffs.append(diff_tmp)

    get_sum = (diff_tmp >= 0.1).sum() 
    if (get_sum > longest_range_for_diff):
      longest_range_for_diff = get_sum

  for i in range(1, length+1):
    tmp = df2[i].values.tolist()
    y_integrals.append(np.trapz(tmp, x_vals))

  durations = []
  n=-1
  for i,j in enumerate(amps):
    amps[i]
    if i%length==0:
      n+=1

    durations.append(times[n])
  
  
  _a_, _c_, _e_ = [], [], []
  concentrations, spinCoating, increasingPPM, temperature, repeat, days, integrals = [], [], [], [], [], [], []

  m=0
  for i in range(len(amps)):

    if(m==len(ppm)): 
      m=0
    concentrations.append(ppm[m])

    spinCoating.append(isSpincoated[m])
    increasingPPM.append(isIncreasing[m])
    temperature.append(temperature_values[m])
    repeat.append(isRepeatUse[m])

    integrals.append(y_integrals[m])
    
    days.append(num_days[m])

    _a_.append(_A_[m])
    _c_.append(_C_[m])
    _e_.append(_E_[m])
    m+=1
  
  dict3 = { 'Time':durations,
            'Current':amps, 
            'Spin Coating':spinCoating ,
            'Increasing PPM':increasingPPM, 
            'Temperature':temperature, 
            'Repeat Sensor Use':repeat, 
            'Days Elapsed':days , 

            'A':_a_,
            'B':_c_,
            'C':_e_,
            'Integrals': integrals,
            'Concentration':concentrations

            }
  df_final = pd.DataFrame(dict3)

  columns = df_final.columns.to_list()

  return df_final


