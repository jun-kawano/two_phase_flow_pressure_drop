import fluids.two_phase
import numpy as np
import CoolProp.CoolProp as CP
from scipy.integrate import simpson
import csv
from kim_mudawar_boiling import *


def get_fluid_properties(fluid, T):
    P_sat = CP.PropsSI('P', 'T', T, 'Q', 0, fluid)  # Pressão de saturação em Pa
    rho_liquid = CP.PropsSI('D', 'T', T, 'Q', 0, fluid)  # Densidade do líquido em kg/m³
    rho_vapor = CP.PropsSI('D', 'T', T, 'Q', 1, fluid)  # Densidade do vapor em kg/m³
    viscosity_liquid = CP.PropsSI('V', 'T', T, 'Q', 0, fluid)  # Viscosidade do líquido em Pa.s
    viscosity_vapor = CP.PropsSI('V', 'T', T, 'Q', 1, fluid)  # Viscosidade do vapor em Pa.s
    sigma = CP.PropsSI('I', 'T', T, 'Q', 0, fluid)  # Tensão superficial em N/m
    h_lv = CP.PropsSI('H', 'T', T, 'Q', 1, fluid) - CP.PropsSI('H', 'T', T, 'Q', 0, fluid)  # Entalpia de vaporização em J/kg
    return (P_sat, rho_liquid, rho_vapor, viscosity_liquid, viscosity_vapor, sigma, h_lv)

def void_fraction_zivi(x, rho_g, rho_f):
    alpha = (1 + ((1 - x) / x) * (rho_g / rho_f) ** (2/3)) ** -1
    return alpha

def momentum_pressure_drop(G, x, rho_g, rho_f):
    result = 0
    void_frac = void_fraction_zivi(x, rho_g=rho_g, rho_f=rho_f)
    a = (x**2) * (1/rho_g) * (1/void_frac)
    b = (1/rho_f) * ((1-x)**2) * (1/(1-void_frac))
    result = (G**2) * (a+b)
    return result

def momentum_dp(G, x_in, x_out, rho_g, rho_f, num=100):
    # Geração de pontos entre x_in e x_out
    x_values = np.linspace(x_in, x_out, num)
    dx = (x_out - x_in)/num
    # Função interna que calcula o valor do gradiente da pressão
    def momentum_pressure_drop_x(x):
        void_frac = void_fraction_zivi(x, rho_g=rho_g, rho_f=rho_f)
        a = (x**2) * (1 / rho_g) * (1 / void_frac)
        b = (1 / rho_f) * ((1 - x)**2) * (1 / (1 - void_frac))
        return (G**2) * (a + b)
    pressure_drop_values = np.array([momentum_pressure_drop_x(x) for x in x_values])
    dpdx = np.gradient(pressure_drop_values, x_values)
    dp = dpdx * dx
    return x_values, dp

def frictional_pressure_drop_muller_steinhagen_heck(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, flow_area=False
):
    if flow_area:
        area = flow_area
    else:
        area = np.pi * (d_h**2) / 4
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = (x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = fluids.two_phase.Muller_Steinhagen_Heck(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            D=d_h,
            L=dz
        )
        dp = dp_frictional
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p


def pressure_drop_muller_steinhagen_heck(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, flow_area=False
):
    if flow_area:
        area = flow_area
    else:
        area = np.pi * (d_h**2) / 4
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = (x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    momentum_dp_values = momentum_dp(G=G, x_in=x_in, x_out=x_out,rho_g=rho_g, rho_f=rho_f, num=num_dots)
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = fluids.two_phase.Muller_Steinhagen_Heck(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            D=d_h,
            L=dz
        )
        dp = dp_frictional + momentum_dp_values[1][i]
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p

def pressure_drop_lockhart_martinelli(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, flow_area=False
):
    if flow_area:
        area = flow_area
    else:
        area = np.pi * (d_h**2) / 4
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = (x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    momentum_dp_values = momentum_dp(G=G, x_in=x_in, x_out=x_out,rho_g=rho_g, rho_f=rho_f, num=num_dots)
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = fluids.two_phase.Lockhart_Martinelli(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            D=d_h,
            L=dz
        )
        dp = dp_frictional + momentum_dp_values[1][i]
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p

def pressure_drop_kim_mudawar_adiabatic(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, flow_area=False
):
    if flow_area:
        area = flow_area
    else:
        area = np.pi * (d_h**2) / 4
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = np.abs(x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    momentum_dp_values = momentum_dp(G=G, x_in=x_in, x_out=x_out,rho_g=rho_g, rho_f=rho_f, num=num_dots)
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = fluids.two_phase.Kim_Mudawar(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            sigma=sigma,
            D=d_h,
            L=dz
        )
        dp = dp_frictional + momentum_dp_values[1][i]
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p

def pressure_drop_kim_mudawar_boiling_backup(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, heated_perimeter, q, flow_area=False
):
    if flow_area:
        area = flow_area
    else:
        area = np.pi * (d_h**2) / 4
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = np.abs(x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    momentum_dp_values = momentum_dp(G=G, x_in=x_in, x_out=x_out,rho_g=rho_g, rho_f=rho_f, num=num_dots)
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = Kim_Mudawar_boiling_backup(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            sigma=sigma,
            D=d_h,
            flow_area=flow_area,
            heated_perimeter=heated_perimeter,
            q=q,
            Hvap=h_lv,
            L=dz
        )
        dp = dp_frictional + momentum_dp_values[1][i]
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p

def pressure_drop_kim_mudawar_boiling(
    fluid, T, m_dot, d_h, L, x_in, x_out, num_dots, heated_perimeter, q, flow_area=False, frictional_only=False
):
    r'''returns (delta_p, dp, z, x, dpdz, dpdx)'''
    if flow_area:
        area = flow_area
    else:
        area = np.pi*d_h*d_h*0.25
    fluid_properties = get_fluid_properties(fluid=fluid, T=T)
    P_sat, rho_f, rho_g, mu_f, mu_g, sigma, h_lv = fluid_properties
    x_values = np.linspace(x_in, x_out, num_dots)
    area = np.pi * (d_h**2) / 4
    G = m_dot / area
    dz = L / num_dots
    dx = np.abs(x_out - x_in) / num_dots
    a = (x_out - x_in) / L
    b = x_in

    pressure_drop_gradient_z = []
    pressure_drop_gradient_x = []
    pressure_drop_sum = 0
    p_values = []
    z_values = []
    i=0
    if frictional_only:
        momentum_dp_values = [0, np.zeros(num_dots)]
    else:
        momentum_dp_values = momentum_dp(G=G, x_in=x_in, x_out=x_out,rho_g=rho_g, rho_f=rho_f, num=num_dots)
    while i < len(x_values):
        x = x_values[i]
        z = (x - b) / a
        dp_frictional = Kim_Mudawar_boiling(
            m=m_dot,
            x=x,
            rhol=rho_f,
            rhog=rho_g,
            mul=mu_f,
            mug=mu_g,
            sigma=sigma,
            Dh=d_h,
            flow_area=flow_area,
            heated_perimeter=heated_perimeter,
            q=q,
            Hvap=h_lv,
            L=dz
        )
        dp = dp_frictional + momentum_dp_values[1][i]
        dp_dz = dp / dz
        dp_dx = dp / dx
        pressure_drop_sum += dp
        pressure_drop_gradient_z.append(dp_dz)
        pressure_drop_gradient_x.append(dp_dx)
        z_values.append(z)
        p_values.append(dp)
        i+=1
    pressure_drop_gradient_z = np.array(pressure_drop_gradient_z)
    pressure_drop_gradient_x = np.array(pressure_drop_gradient_x)
    delta_p = (
        pressure_drop_sum, p_values, z_values, x_values, pressure_drop_gradient_z, pressure_drop_gradient_x
    )
    return delta_p

if __name__ == '__main__':#================================================

    from scipy.integrate import simpson
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    d_h = 1.49E-3#0.01092
    micro_channel_length = 0.398#3.013
    m_dot = 0.0194  # kg/s
    x_in = 0.1
    x_out = 0.9
    T = 4 + 273.15  # 0°C em Kelvin
    fluido = 'R134a'
    area = np.pi * (d_h**2) *0.25
    n=100
    
    delta_p1 = pressure_drop_muller_steinhagen_heck(T=T,
                                                fluid=fluido,
                                                x_in=x_in,
                                                x_out=x_out,
                                                m_dot=m_dot,
                                                d_h=d_h,
                                                L=micro_channel_length,
                                                num_dots=n
                                                )
    # Calcular a queda de pressão usando o modelo Lockhart-Martinelli
    delta_p2 = pressure_drop_lockhart_martinelli(T=T,
                                                fluid=fluido,
                                                x_in=x_in,
                                                x_out=x_out,
                                                m_dot=m_dot,
                                                d_h=d_h,
                                                L=micro_channel_length,
                                                num_dots=n
                                                )
    
    delta_p3 = pressure_drop_kim_mudawar_adiabatic(fluid=fluido,
                                                 T=T,
                                                 m_dot=m_dot,
                                                 d_h=d_h,
                                                 L=micro_channel_length,
                                                 x_in=x_in,
                                                 x_out=x_out,
                                                 num_dots=n,
                                                 )
    delta_p4 = pressure_drop_kim_mudawar_boiling(fluid=fluido,
                                                 T=T,
                                                 m_dot=m_dot,
                                                 d_h=d_h,
                                                 L=micro_channel_length,
                                                 x_in=x_in,
                                                 x_out=x_out,
                                                 num_dots=n,
                                                 heated_perimeter=np.pi*d_h,
                                                 q=1000,
                                                 flow_area=area
                                                 )
    delta_p5 = pressure_drop_kim_mudawar_boiling_backup(fluid=fluido,
                                                 T=T,
                                                 m_dot=m_dot,
                                                 d_h=d_h,
                                                 L=micro_channel_length,
                                                 x_in=x_in,
                                                 x_out=x_out,
                                                 num_dots=n,
                                                 heated_perimeter=np.pi*d_h,
                                                 q=1000,
                                                 flow_area=area
                                                 )

    print('========')
    print(f'DELTA_P Muller-Steinhagen-Heck  : {(delta_p1[0])/1000:.2f} kPa')
    print(f'DELTA_P Lockhart-Martinelli     : {(delta_p2[0])/1000:.2f} kPa')
    print(f'DELTA_P Kim-Mudawar Adiabatic   : {(delta_p3[0])/1000:.2f} kPa')
    print(f'DELTA_P Kim-Mudawar Boiling     : {(delta_p4[0])/1000:.2f} kPa')
    #print(f'DELTA_P Kim-Mudawar Boiling 2   : {(delta_p5[0])/1000:.2f} kPa')
    print('========')




