# GraphPoint-Extractor
Performed image processing techniques to extract data points from graph plot.

# Description
This process began with image processing techniques, including loading the image in grayscale, applying GaussianBlur to reduce noise, and using Canny edge detection to identify edges. Contour detection was then employed to isolate the line representing the data points. The largest contour, assumed to represent the data line, was extracted and refined with the Douglas-Peucker algorithm to approximate it with additional points. Subsequently, the image coordinates were converted to data coordinates, providing the x and y points for the graph. To store and visualize this data, a Pandas DataFrame was created, storing the data points and saving them to a CSV file for future use. Finally, Plotly was utilized to create an interactive line graph from the extracted data points, enabling users to explore and analyze the data visually.

