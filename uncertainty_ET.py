import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
import io

# set title
#st.set_page_config(layout="wide")

st.title('Uncertainty in triangle zones')
st.markdown('''This is a simple visualization of uncertainties in the model presented by Sofia Brisson. 
It contains options to view either probabilities or information entropy. These were calculated using the open source 
modelling library **Gempy**. Results can be seen on my github''')
st.radio("Uncertainty style (does not work yet)",
         ('Probability', 'Entropy'))


# set data source

DATA_URL = 'https://raw.githubusercontent.com/NilsChudalla/Test_Sofia/main/entropy_block.csv'


# load data into cache
@st.cache
def load_data():
    block = pd.read_csv(DATA_URL, usecols=['vals']).values.reshape(50,50,50)
    #block = np.load(DATA_URL, allow_pickle=True).reshape(50, 50, 50)
    return block
# info text for loading data
data_load_state = st.text('Loading data...')
block = load_data()
data_load_state.text("Done! (using st.cache)")

# Setting plotting parameters
xmin, xmax, ymin, ymax, zmin, zmax = [6, 25, 3, 35, -5.5, 0.5]
resolution = np.array([50.0, 50.0, 50.0])
d_grid = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
cell_width = d_grid / resolution
grid_min = np.array([xmin, ymin, zmin]) + cell_width/0.5
grid_max = np.array([xmax, ymax, zmax]) - cell_width/0.5

x_array = np.linspace(grid_min[0], grid_max[0], resolution[0].astype(int))
y_array = np.linspace(grid_min[1], grid_max[1], resolution[1].astype(int))
z_array = np.linspace(grid_min[2], grid_max[2], resolution[2].astype(int))

XX_xy, YY_xy = np.meshgrid(x_array, y_array)
XX_xz, ZZ_xz = np.meshgrid(x_array, z_array)
YY_yz, ZZ_yz = np.meshgrid(y_array, z_array)

# Add slider

st.subheader('Uncertainty maps')
with st.expander('Profile sliders'):

    depth = st.slider(label='Horizontal level',min_value=float(np.min(z_array)),max_value=float(np.max(z_array)), value=float(np.min(z_array)))
    WE_profile = st.slider(label='W-E profile',min_value=float(np.min(y_array)),max_value=float(np.max(y_array)), value=float(np.min(y_array)))
    NS_profile = st.slider(label='N-S profile',min_value=float(np.min(x_array)),max_value=float(np.max(x_array)), value=float(np.min(x_array)))

cross_sections = st.checkbox('Toggle relative profile positions')

depth_index = np.argmin(np.abs(z_array-depth))
WE_index = np.argmin(np.abs(y_array-WE_profile))
NS_index = np.argmin(np.abs(x_array-NS_profile))

hslice_coords = block[:,:,depth_index]
WE_slice = block[:, WE_index, :]
NS_slice = block[NS_index,:,:]

# Create subsection
val1, val2 = st.select_slider('Select an entropy range',options=np.linspace(0,2,10).astype('<3U'), value=('0.0', '2.0'))
vmin = np.min(np.array([float(val1), float(val2)]))
vmax = np.max(np.array([float(val1), float(val2)]))

fig = plt.figure()
AX = gridspec.GridSpec(2,2)
AX.update(wspace = 0.5, hspace = 0.5)
ax1 = plt.subplot(AX[0,0])
ax1.set_title('Depthmap uncertainty')
ax1.contourf(XX_xy, YY_xy, hslice_coords, cmap='magma', vmin=vmin, vmax=vmax)
ax1.axis('equal')
ax1.set_xlim(np.min(x_array), np.max(x_array))
ax1.set_ylim(np.min(y_array), np.max(y_array))

if cross_sections:
    ax1.vlines(NS_profile, ymin=grid_min[1], ymax=grid_max[1], color='white', lw=1)
    ax1.hlines(WE_profile, xmin=grid_min[0], xmax=grid_max[0], color='white', lw=1)

ax2 = plt.subplot(AX[0,1])
ax2.set_title('Uncertainty - W - E')
ax2.contourf(XX_xz, ZZ_xz, WE_slice, cmap='magma', vmin=vmin, vmax=vmax)
ax2.axis('equal')
ax2.set_xlim(np.min(x_array), np.max(x_array))
ax2.set_ylim(np.min(z_array), np.max(z_array))

if cross_sections:
    ax2.vlines(NS_profile, ymin=grid_min[2], ymax=grid_max[2], color='white', lw=1)
    ax2.hlines(depth, xmin=grid_min[0], xmax=grid_max[0], color='white', lw=1)

ax3 = plt.subplot(AX[1,0])
ax3.set_title('Uncertainty - S - N')
im = ax3.contourf(YY_yz, ZZ_yz, NS_slice, cmap='magma', vmin=vmin, vmax=vmax)
ax3.axis('equal')
ax3.set_xlim(np.min(y_array), np.max(y_array))
ax3.set_ylim(np.min(z_array), np.max(z_array))

if cross_sections:
    ax3.vlines(WE_profile, ymin=grid_min[2], ymax=grid_max[2], color='white', lw=1)
    ax3.hlines(depth, xmin=grid_min[1], xmax=grid_max[1], color='white', lw=1)

ax4 = fig.add_axes([0.65, 0.05, 0.05, 0.37])
x1 = np.array([0.0,1.0])
XX1, ZZ1 = np.meshgrid(x1, z_array)
vals = np.vstack((block[NS_index, WE_index,:], block[NS_index, WE_index,:]))
ax4.set_title('"Borehole view"')
ax4.contourf(XX1, ZZ1, vals.T, cmap='magma', vmin=vmin, vmax=vmax)
ax4.set_xlim(0.1, 0.9)
ax4.axes.get_xaxis().set_ticks([])

if cross_sections:
    ax4.hlines(depth, xmin=-1, xmax=2, color='white', lw=1)

ax5 = fig.add_axes([0.8, 0.05, 0.05, 0.36])
fig.colorbar(im, cax=ax5, orientation='vertical', ticks=np.linspace(vmin, vmax, 4), boundaries=[vmin, vmax], label='entropy')


st.pyplot(fig)


