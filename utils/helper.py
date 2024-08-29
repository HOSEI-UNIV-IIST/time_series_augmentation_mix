import os

import numpy as np
import plotly.graph_objs as go

def plot2d(x, y, x2=None, y2=None, x3=None, y3=None, xlim=(-1, 1), ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2)
    if x3 is not None and y3 is not None:
        plt.plot(x3, y3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return

def plot1d(x, x2=None, x3=None, ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    steps = np.arange(x.shape[0])
    plt.plot(steps, x)
    if x2 is not None:
        plt.plot(steps, x2)
    if x3 is not None:
        plt.plot(steps, x3)
    plt.xlim(0, x.shape[0])
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return


def save_x_train(x_train, name, extension="npy"):
    # Ensure the 'data' directory exists
    os.makedirs('./data', exist_ok=True)

    # Determine the file path
    file_path = os.path.join('./data', f"{name}_train.{extension}")

    # Save the file in the desired format
    if extension == "npy":
        np.save(file_path, x_train)
    elif extension == "csv":
        # Reshape for saving in CSV
        reshaped_x_train = x_train.reshape(x_train.shape[0], -1)
        np.savetxt(file_path, reshaped_x_train, delimiter=",")
    else:
        raise ValueError("Unsupported file extension. Use 'npy' or 'csv'.")

    print(f"x_train saved to {file_path}")


def load_x_train(name, extension="npy"):
    # Determine the file path
    file_path = os.path.join('./data', f"{name}_train.{extension}")

    # Load the file based on the extension
    if extension == "npy":
        x_train = np.load(file_path)
    elif extension == "csv":
        x_train = np.loadtxt(file_path, delimiter=",")
    else:
        raise ValueError("Unsupported file extension. Use 'npy' or 'csv'.")

    print(f"x_train loaded from {file_path}")
    return x_train




def plot1d_plotly(x, x2=None, x3=None, label1='original', label2='x2', label3='x3', ylim=(-1, 1), save_file=''):
    # Create steps for the x-axis
    steps = np.arange(x.shape[0])

    # Create the plot using Plotly
    fig = go.Figure()

    # Add the first trace with its label
    fig.add_trace(go.Scatter(x=steps, y=x, mode='lines', name=label1))

    # Add the second trace with its legend (labeled as "label2 transformation")
    if x2 is not None:
        fig.add_trace(go.Scatter(x=steps, y=x2, mode='lines', name=label2))

    # Add the third trace with its label
    if x3 is not None:
        fig.add_trace(go.Scatter(x=steps, y=x3, mode='lines', name=label3))

    # Update layout with visible axes and a white background
    fig.update_layout(
        xaxis=dict(
            range=[0, x.shape[0]],
            showline=True,  # Ensure the x-axis line is visible
            showgrid=False,  # Optionally disable grid lines
            zeroline=False,  # Optionally disable the zero line
            linecolor='black',  # Make sure the axis line is black
            tickcolor='black',  # Make sure the tick marks are black
            tickfont=dict(color='black')  # Make sure the tick labels are black
        ),
        yaxis=dict(
            range=ylim,
            showline=True,  # Ensure the y-axis line is visible
            showgrid=False,  # Optionally disable grid lines
            zeroline=False,  # Optionally disable the zero line
            linecolor='black',  # Make sure the axis line is black
            tickcolor='black',  # Make sure the tick marks are black
            tickfont=dict(color='black')  # Make sure the tick labels are black
        ),
        width=600,  # Increased width for higher resolution
        height=300,  # Increased height for higher resolution
        margin=dict(l=40, r=40, t=40, b=100),  # Adjusted bottom margin for legend
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Adjust this to move the legend below the plot
            xanchor="center",
            x=0.5,
            font=dict(color="black"),
        ),
        plot_bgcolor='white',  # White plot background
        paper_bgcolor='white',  # White paper background
    )

    # Save or show the figure
    if save_file:
        plot_base_path = "./plots/"
        file_name = f'{label2}.eps'
        os.makedirs('./plots', exist_ok=True)
        path = os.path.join(plot_base_path, save_file, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the figure as an EPS file
        fig.write_image(path, format='eps')
        print(f"Plot saved as {path}")
    else:
        fig.show()

    return
