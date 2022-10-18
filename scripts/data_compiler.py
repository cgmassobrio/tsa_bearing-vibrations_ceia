# Librerías principales
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# Manipulación de archivos
import os
import glob

# Pruebas de tiempos de ejecución
import time

# Versiones de librerías
print("".join(f"{x[0]}: {x[1]}\n" for x in [
    ("Numpy",np.__version__),
    ("Pandas",pd.__version__),
    ("Matplotlib",matplotlib.__version__),
]))

# Definición de la ruta
path_record = './data/record'
path_data = './data'


# Definición del nombre de columnas
colnames = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

# Armado del dataset
all_files = glob.glob(os.path.join(path_record, "*.39"))
df_from_each_file = (pd.read_csv(f, sep='\t', names=colnames, header=None, dtype='float32') for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)

# frecuencia de muestreo o "sampling rate" (Hz)
sr = 20e3
# tiempo total de muestreo
tt = np.around(len(df)/sr, decimals=5)
# array con los tiempos de muestreo
time_sample = np.r_[0:tt:1/sr].astype('float32')

# Incorporación de columna con tiempos de muestreo
df.insert(0, 'Time', time_sample)


# # Gráfico de vibraciones en rodamientos
# fig = plt.figure(figsize=(18,6))
# for i in range(4):
#     plt.plot(df['Time'], df.iloc[:,i])
# plt.xlim(-1, df['Time'].iloc[-1]+1)
# plt.ylabel('desplazamiento ($\mu$m)')
# plt.xlabel('tiempo de muestreo (s)')
# plt.legend(['Rodamiento 1', 'Rodamiento 2', 'Rodamiento 3', 'Rodamiento 4'])
# plt.title('Vibraciones sensadas en rodamientos', fontsize=14)

# plt.gca().yaxis.grid(True)
# plt.show()

# Generación del dataset
df.to_csv(os.path.join(path_record, "dataset.csv"), index=False)