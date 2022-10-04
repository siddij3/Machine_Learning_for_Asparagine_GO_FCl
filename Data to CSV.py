# In[12]:


import csv
import pandas as pd
import os


# In[13]:


def transform_data(csvfile):

  df = pd.read_csv(csvfile, header=None)

  isSpincoated = df.values.tolist()[0][1:]
  isIncreasing = df.values.tolist()[1][1:]
  temperature_values = df.values.tolist()[2][1:]
  isRepeatUse = df.values.tolist()[3][1:]
  num_days = df.values.tolist()[4][1:]



  fcl_values = df.values.tolist()[5]
  fcl_values = [x for x in fcl_values if str(x) != 'nan' and 'omit' not in str(x).lower()]


  ppm = []

  for i in fcl_values:
    try:
      ppm.append(float(i))
    except:
      pass

  exp_params = df.values.tolist()
  params = []

  for i in range (0, 5):
      params.append(exp_params[i][0])

  
  df2 = df[6:].dropna(axis='columns')
  
  print(df2)

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


  durations = []
  n=-1
  for i in range(len(amps)):
    amps[i]
    if i%length==0:
      n+=1

    durations.append(times[n])
  
  concentrations = []
  spinCoating = []
  increasingPPM = []
  temperature = []
  repeat = []
  days = []

  m=0
  for i in range(len(amps)):
    if(m==len(ppm)): 
      m=0
    concentrations.append(ppm[m])

    spinCoating.append(isSpincoated[m])
    increasingPPM.append(isIncreasing[m])
    temperature.append(temperature_values[m])
    repeat.append(isRepeatUse[m])

    days.append(num_days[m])
    m+=1
  
  dict3 = {'Time':durations,'Current':amps, 'Spin Coating':spinCoating ,'Increasing PPM':increasingPPM, 'Temperature':temperature, 'Repeat Sensor Use':repeat, 'Days Elapsed':days , 'Concentration':concentrations}
  df_final = pd.DataFrame(dict3)

  columns = df_final.columns.to_list()

  return df_final


# In[14]:


filepath = r"C:\\Users\\junai\\Documents\\McMaster\\Food Packaging\\Thesis\\Thesis Manuscript\\Experiments\\Raw Data\\ML Parsed"
local_download_path = os.path.expanduser(filepath)
filenames=[]
for filename in os.listdir(local_download_path):
    if filename.endswith('csv') and "Entries" in filename:
      filenames.append(filepath + "\\" + filename)

sum = 0
for i in range(len(filenames)):

  df = transform_data(filenames[i])

  sum+=len(df)
  
if len(filenames)>0:
  for i in filenames[0:]:
    #print('File appended: '+i)
    df= df.append(transform_data(i),ignore_index=True,sort=False)
df.to_csv('aggregated_data2.csv',index=False)


# In[15]:


import numpy as np
for i in df.to_numpy():
    if (np.isnan(i).any()):
        print(i)

