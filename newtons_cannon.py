import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def simple_projectile(u, deg, ax):
    """""
    Simulates projectile motion over a flat surface.

    Parameters:
        u (float): Initial speed in m/s.
        deg (float): Launch angle in degrees.
        ax: Axes object to plot trajectory.

    """

    deg_rad = np.radians(deg) # convert degrees to radians
    g = 9.81 # gravitational acceleration

    t_flight = 2 * u * np.sin(deg_rad) / g # time until projectile hits the ground
    t = np.linspace(0, t_flight, 1000) 
    
    x = u * np.cos(deg_rad) * t    # horizontal distance
    y = u * np.sin(deg_rad) * t - 0.5 * g * t**2 # vertical distance
    
    ax.plot(x, y, label=f"u={u} m/s, θ={deg}°")  


def orbital_velocity(h):
    """
    Computes orbital velocity at altitude h above Earth's surface.

    Parameters:
        h (float): Altitude in meters.

    Returns:
        float: Orbital velocity in m/s.
    """

    R_E = 6371000      
    GM = 3.986e14 
    r = R_E + h
    return np.sqrt(GM / r)


def escape_velocity(h):
    """
    Computes escape velocity at altitude h above Earth's surface.

    Parameters:
        h (float): Altitude in meters.

    Returns:
        float: Escape velocity (m/s).
    """

    R_E = 6371000      
    GM = 3.986e14 
    r = R_E + h
    return np.sqrt(2* GM / r)

def newtons_cannon(factor, deg, h, ax, t_max, label):
    """
    Simulates Newton's cannonball thought experiment. 

    Parameters:
        factor (float): Multiple of orbital velocity, scaling initial velocity. 
        deg (float): Lauch angle in degrees. 
        h (float): Altitude above Earth's surface in km. 
        ax: Axes object to plot trajectory.
        t_max (float): Max simulation time in seconcs. 
        label (str): Label for the plot.
    
    """
    R_E = 6371000      
    GM = 3.986e14        
    r0 = R_E + h * 1000  # initial distance from Earth's center in m
    u = factor * np.sqrt(GM / r0)  # orbit velocity multiplied by factor

    deg_rad = np.radians(deg) # convert angle into radians
    ux = u * np.cos(deg_rad) # resolve initial velocity 
    uy = u * np.sin(deg_rad)

    init_cond = [0, r0, ux, uy]  # initial conditions: [x, y, vx, vy]

    def sim(t, Y):  # function which simulates equations of motion
        x, y, vx, vy = Y
        r = np.sqrt(x**2 + y**2)
        ax = -GM * x / r**3
        ay = -GM * y / r**3
        return [vx, vy, ax, ay]

    
    out = solve_ivp(sim, [0, t_max], init_cond, max_step=1) # ODE solver - determines the differential equations from initial values and inputted function

    x, y = out.y[0], out.y[1]
    r = np.sqrt(x**2 + y**2)

    # hit condition to stop once projectile hits the ground
    hit = np.where(r < R_E)[0] # return indices where r < R_E
    if hit.size > 0:
        x, y = x[:hit[0]], y[:hit[0]] # truncate after impact


    ax.plot(x, y, label=label) # plot x and y coordinates


def newtons_cannon_drag(factor, deg, h, ax, t_max, label, C_d=0.47, rho=1.225, A=0.01, mass=1000.0):
    
    """
    Simulates Newton's cannon with constant atmospheric drag.

    Parameters:
        factor (float): Multiple of orbital velocity.
        deg (float): Launch angle in degrees.
        h (float): Launch altitude in km.
        ax: Axes object to plot trajectory.
        t_max (float): Max simulation time in seconds.
        label (str): Label for plot.
        C_d (float): Drag coefficient.
        rho (float): Air density in kg/m^3.
        A (float): Cross-sectional area in m^2.
        mass (float): Mass of projectile in kg.
    """

    
    
    R_E = 6371000      
    GM = 3.986e14        
    r0 = R_E + h*1000   
    u = factor * np.sqrt(GM / r0) 

    deg_rad = np.radians(deg)
    ux = u * np.cos(deg_rad)  
    uy = u * np.sin(deg_rad)  

    init_cond = [0, r0, ux, uy] 

    def sim_drag(t, Y, C_d, rho, A, mass): 
        x, y, vx, vy = Y
        r = np.sqrt(x**2 + y**2)
        v = np.sqrt(vx**2 + vy**2)

        # acceleration due to gravity
        ax_g = -GM * x / r**3
        ay_g = -GM * y / r**3

        # acceleration due to drag
        ax_d = -0.5 * C_d * rho * A * v * vx / mass
        ay_d = -0.5 * C_d * rho * A * v * vy / mass

        # total acceleration
        ax = ax_g + ax_d
        ay = ay_g + ay_d

        return [vx, vy, ax, ay]

             
    out = solve_ivp(sim_drag, [0, t_max], init_cond, args=(C_d, rho, A, mass), max_step=1)

    x, y = out.y[0], out.y[1]
    r = np.sqrt(x**2 + y**2)

   
    hit = np.where(r < R_E)[0] 
    if hit.size > 0:   
        x, y = x[:hit[0]], y[:hit[0]]  
    
    
    ax.plot(x,y, label=label)

   


def newtons_cannon_drag_atmo(factor, deg, h, ax, t_max, label, C_d=0.47, rho0=1.225, A=0.01, mass=100.0):
    """
    Simulates Newton's cannon with altitude-dependent atmospheric drag.

    Parameters:
        factor (float): Multiple of orbital velocity.
        deg (float): Launch angle in degrees.
        h (float): Launch altitude in km.
        ax: Axes object to plot trajectory.
        t_max (float): Max simulation time in seconds.
        label (str): Label for plot.
        C_d (float): Drag coefficient.
        rho0 (float): Air density at sea level in kg/m^3.
        A (float): Cross-sectional area in m^2.
        mass (float): Mass of projectile in kg.
    """
    
    
    R_E = 6_371_000      
    GM = 3.986e14        
    r0 = R_E + h*1000  
    u = factor * np.sqrt(GM / r0)  

    
    deg_rad = np.radians(deg)
    ux = u * np.cos(deg_rad)  
    uy = u * np.sin(deg_rad)  

    init_cond = [0, r0, ux, uy]

    def sim_drag(t, Y, C_d, rho, A, mass): 
        x, y, vx, vy = Y
        r = np.sqrt(x**2 + y**2)
        v = np.sqrt(vx**2 + vy**2)
        alt = max(0, r - R_E)    
        
        # acceleration due to gravity
        ax_g = -GM * x / r**3
        ay_g = -GM * y / r**3

        rho = rho0 * np.exp(-alt / 8500)  # adjust density based on altitude

        # acceleration due to drag
        ax_d = -0.5 * C_d * rho * A * v * vx / mass
        ay_d = -0.5 * C_d * rho * A * v * vy / mass

        # total acceleration
        ax = ax_g + ax_d
        ay = ay_g + ay_d

        return [vx, vy, ax, ay]
    
             
    out = solve_ivp(sim_drag, [0, t_max], init_cond, args=(C_d, rho0, A, mass), max_step=1)

    x, y = out.y[0], out.y[1]
    r = np.sqrt(x**2 + y**2)

    hit = np.where(r < R_E)[0] 
    if hit.size > 0:   
        x, y = x[:hit[0]], y[:hit[0]] 
    
    ax.plot(x, y, label=label)

    