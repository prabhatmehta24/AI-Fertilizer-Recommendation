import os
import pandas as pd

path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
df = pd.read_csv(path)
cols = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous','Soil_Type','Crop_Type']
print('columns:', [c for c in cols if c in df.columns])
print(df[cols].head(5).to_string(index=False))
print('\nmin/max:')
for c in ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']:
    if c in df.columns:
        print(c, df[c].min(), df[c].max())
print('\nsoil types:', df['Soil_Type'].dropna().unique()[:20].tolist())
print('crop types:', df['Crop_Type'].dropna().unique()[:20].tolist())
