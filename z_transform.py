import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Symbol, exp

def get_z_transform(sequence_type, *params):
    """Calculate Z-transform for different sequences"""
    z = Symbol('z')

    if sequence_type == "exponential":
        a = params[0]
        transform = z / (z - a)
        roc = f"|z| > {abs(a)}"
    elif sequence_type == "sum_exponential":
        a1, a2 = params
        transform = z/(z - a1) + z/(z - a2)
        roc = f"|z| > max({abs(a1)}, {abs(a2)})"
    elif sequence_type == "sinusoidal":
        w = params[0]
        transform = z * np.sin(w) / (z**2 - 2*z*np.cos(w) + 1)
        roc = "|z| > 1"

    return transform, roc

def plot_roc(roc_condition):
    """Simple ROC plot"""
    # Extract radius value from ROC condition
    if 'max' in roc_condition:
        # Handle case with max() in ROC
        numbers = [float(x.strip()) for x in roc_condition.split('>')[-1].strip('max() ').split(',')]
        radius = max(numbers)
    else:
        # Handle simple case
        radius = float(roc_condition.split('>')[-1].strip())

    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    plt.figure(figsize=(4, 4))
    plt.plot(x, y, 'r--', label='ROC Boundary')
    plt.grid(True)
    plt.axis('equal')
    plt.title(f'ROC: {roc_condition}')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

def plot_frequency_response(z_transform, show_magnitude=True, show_phase=True):
    """
    Plot frequency response with toggleable magnitude and phase
    """
    theta = np.linspace(0, 2*np.pi, 500)
    z_values = np.exp(1j * theta)
    z = Symbol('z')

    response = [complex(z_transform.subs(z, z_val)) for z_val in z_values]
    magnitude = np.abs(response)
    phase = np.angle(response)

    if show_magnitude or show_phase:
        if show_magnitude and show_phase:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            ax1.plot(theta, magnitude)
            ax1.set_title('Magnitude Response')
            ax1.set_xlabel('Frequency (radians)')
            ax1.set_ylabel('Magnitude')
            ax1.grid(True)

            ax2.plot(theta, phase)
            ax2.set_title('Phase Response')
            ax2.set_xlabel('Frequency (radians)')
            ax2.set_ylabel('Phase (radians)')
            ax2.grid(True)

            plt.tight_layout()

        elif show_magnitude:
            plt.figure(figsize=(4, 4))
            plt.plot(theta, magnitude)
            plt.title('Magnitude Response')
            plt.xlabel('Frequency (radians)')
            plt.ylabel('Magnitude')
            plt.grid(True)

        else:  # show_phase only
            plt.figure(figsize=(6, 4))
            plt.plot(theta, phase)
            plt.title('Phase Response')
            plt.xlabel('Frequency (radians)')
            plt.ylabel('Phase (radians)')
            plt.grid(True)

        plt.show()

# Solve the given sequences
# 1. 3^n u[n]
print("\n1. Sequence: 3^n u[n]")
transform, roc = get_z_transform("exponential", 3)
print(f"Z-transform: {transform}")
print(f"ROC: {roc}")
plot_roc(roc)
plot_frequency_response(transform, show_magnitude=False, show_phase=False)

# 2. 2^n*u[n] + 3^n*u[n]
print("\n2. Sequence: 2^n*u[n] + 3^n*u[n]")
transform, roc = get_z_transform("sum_exponential", 2, 3)
print(f"Z-transform: {transform}")
print(f"ROC: {roc}")
plot_roc(roc)
plot_frequency_response(transform, show_magnitude=False, show_phase=False)

# 3. e^(-0.2n)*u[n]
print("\n3. Sequence: e^(-0.2n)*u[n]")
transform, roc = get_z_transform("exponential", np.exp(-0.2))
print(f"Z-transform: {transform}")
print(f"ROC: {roc}")
plot_roc(roc)
plot_frequency_response(transform, show_magnitude=True, show_phase=True)

# 4. sin(πn/4)u[n]
print("\n4. Sequence: sin(πn/4)u[n]")
transform, roc = get_z_transform("sinusoidal", np.pi/4)
print(f"Z-transform: {transform}")
print(f"ROC: {roc}")
plot_roc(roc)
plot_frequency_response(transform, show_magnitude=True, show_phase=True)
