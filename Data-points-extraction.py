#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.express as px


# In[2]:


# Load the image in grayscale
image = cv2.imread('/Users/rithikaannareddy/Desktop/CME_Problem-1-graph.jpg', cv2.IMREAD_GRAYSCALE)


# In[3]:


# Apply GaussianBlur to reduce noise
# Applied GaussianBlur to reduce noise in the image. It filters the image with 5*5 Gaussian with 0 sigma. 
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)


# In[4]:


# Apply Canny edge detection
# Parameters to Adjust: low_thresh and high_thresh: Adjusting these can affect the sensitivity of edge detection. apertureSize: The size of the Sobel kernel used for gradient calculation. 
# This default size of 3x3 is commonly used for standard Sobel edge detection, as it captures immediate neighboring pixel values for gradient calculations. 
edges = cv2.Canny(image_blurred, 50, 150, apertureSize=3)


# In[5]:


# Find contours
# RETR_EXTERNAL flag ensures that only the outermost contours are detected
# CHAIN_APPROX_SIMPLE method reduces the number of points required to represent the contours. For example, a straight line will be approximated by only its two endpoints. 
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# In[6]:


# Get the largest contour (assuming it's the line)
largest_contour = max(contours, key=cv2.contourArea)


# In[7]:


# Approximate the contour
# This approximation reduces the number of points in the contour while preserving its shape, simplifying the representation while retaining the overall shape. 
epsilon = 0.001 * cv2.arcLength(largest_contour, True)  # Increased epsilon for more points
approx = cv2.approxPolyDP(largest_contour, epsilon, True)


# In[8]:


# Extract points from the contour
x_points = []
y_points = []
for point in approx:
    x_points.append(point[0][0])
    y_points.append(point[0][1])


# In[9]:


# Define the range of x and y axes
x_min, x_max = 2015.008804, 2023.870207
y_min, y_max = 68.33, 320.854386


# In[10]:


# Convert image coordinates to data coordinates
# The formula uses linear scaling and shifting each coordinate to perform the conversion.
converted_x_points = [x_min + (x_max - x_min) * (x - min(x_points)) / (max(x_points) - min(x_points)) for x in x_points]
converted_y_points = [y_max - (y_max - y_min) * (y - min(y_points)) / (max(y_points) - min(y_points)) for y in y_points]


# In[11]:


# Create a Pandas DataFrame to store the data points
data = {'x': converted_x_points, 'y': converted_y_points}
df = pd.DataFrame(data)

# Print extracted data points
print("Extracted Data Points:")
print(df)

# Save data points to a CSV file
df.to_csv('/Users/rithikaannareddy/Desktop/Default Dataset.csv', index=False)


# In[12]:


# Plot the DataFrame 'df'
plt.scatter(df['x'], df['y'], color='green', marker='x', label='Data Points from DataFrame')
plt.plot(df['x'], df['y'], color='orange', linewidth=2, label='Line from DataFrame')

plt.legend()

plt.show()


# In[13]:


data= pd.read_csv("/Users/rithikaannareddy/Desktop/Default Dataset.csv")
column_names = ["x", "y"]  
data.columns = column_names
data 


# In[14]:


# Round 'x' column to the nearest tenth
data['x'] = np.round(data['x'], 2)
data['y'] = np.round(data['y'], 2)
data


# In[15]:


fig = px.line(data,x ='x',y ='y', color_discrete_sequence=["#19aae3"])
fig.update_traces(line={'width': 3})
fig.update_layout(
    xaxis=dict(
        tickformat='%Y-%m'  # Format for hover and display
    )
)


# In[16]:


# Problem 2
fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(
        range=[2015, 2024],
        tickvals=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],  # Specify tick values
        ticktext=['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],  # Specify tick labels
        showline=True,
        linecolor='lightgrey',  # Line color
        linewidth=1,  # Line width
        tickangle=-45,
    ),
    yaxis=dict(
        range=[0, 350],
        gridcolor='lightgrey',
        tickvals=[0, 50, 100, 150, 200, 250, 300, 350],  # Specify tick values
        ticktext=['0', '50', '100', '150', '200', '250', '300', '350'],  # Specify tick labels
        title="Rig count",
        showline=True,
    ),
    title={
        'text': "<b>Active Natural Gas Rig Counts</b>",
        'x': 0.5,  # Center of the plot
        'y': 0.95,  # Just above the plot
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 16},
    }
)
fig.show()


# In[17]:


# Save the figure as a PNG file
fig.write_html("/Users/rithikaannareddy/Desktop/check.html")


# In[ ]:




