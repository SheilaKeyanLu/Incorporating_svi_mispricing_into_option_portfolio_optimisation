import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
def calc_d1(F,T, sigma, K):
    return (np.log(F / K) + (0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

def calc_d2(F,T, sigma, K):
    return calc_d1(F,T, sigma, K) - sigma*np.sqrt(T)


def calc_call_price(F, discount_factor,T, sigma, K):
    d1 = calc_d1(F,T, sigma, K)
    d2 = calc_d2(F,T, sigma, K)
    call_price = discount_factor * F * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)
    return call_price

def calc_put_price(F, discount_factor,T, sigma, K):
    d1 = calc_d1(F,T, sigma, K)
    d2 = calc_d2(F,T, sigma, K)
    put_price = discount_factor * K * norm.cdf(-d2) - F * discount_factor * norm.cdf(-d1)
    return put_price
def calc_delta(F, discount_factor,T, sigma, K, otpion_type):
    d1 = calc_d1(F,T, sigma, K)
    option_type_arr = np.asarray(otpion_type)
    call_delta = discount_factor * norm.cdf(d1)
    put_delta = -discount_factor * norm.cdf(-(d1))
    delta = np.where(
        option_type_arr == "C",
        call_delta,
        np.where(option_type_arr == "P", put_delta, np.nan),
    )
    return delta

def calc_gamma(F, discount_factor,T, sigma, K):
    d1 = calc_d1(F,T, sigma, K)
    gamma = discount_factor*norm.pdf(d1)/(F*sigma*np.sqrt(T))
    return gamma


def calc_theta(F, discount_factor,T, sigma, K, otpion_type, r):
    d1 = calc_d1(F, T, sigma, K)
    call_price = calc_call_price(F, discount_factor,T, sigma, K)
    put_price = calc_put_price(F, discount_factor,T, sigma, K)
    option_type_arr = np.asarray(otpion_type)
    time_decay = -discount_factor*F*norm.pdf(d1)*sigma/(2*np.sqrt(T))
    call_theta = time_decay+r*call_price
    put_theta = time_decay+r*put_price
    theta = np.where(
        option_type_arr == "C",
        call_theta,
        np.where(option_type_arr == "P", put_theta, np.nan),
    )
    return theta

def calc_vega(F, discount_factor,T, sigma, K):
    d1 = calc_d1(F, T, sigma, K)
    vega = F*discount_factor*np.sqrt(T)*norm.pdf(d1)
    return vega


def calc_spot_d1(S, T, sigma, K, r):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def calc_spot_d2(S, T, sigma, K, r):
    return calc_spot_d1(S, T, sigma, K, r) - sigma * np.sqrt(T)


def calc_spot_call_price(S, T, sigma, K, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    d2 = calc_spot_d2(S, T, sigma, K, r)
    discount_factor = np.exp(-r * T)
    return S * norm.cdf(d1) - K * discount_factor * norm.cdf(d2)


def calc_spot_put_price(S, T, sigma, K, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    d2 = calc_spot_d2(S, T, sigma, K, r)
    discount_factor = np.exp(-r * T)
    return K * discount_factor * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_spot_delta(S, T, sigma, K, option_type, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    option_type_arr = np.asarray(option_type)
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1
    return np.where(
        option_type_arr == "C",
        call_delta,
        np.where(option_type_arr == "P", put_delta, np.nan),
    )


def calc_spot_gamma(S, T, sigma, K, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calc_spot_theta(S, T, sigma, K, option_type, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    d2 = calc_spot_d2(S, T, sigma, K, r)
    option_type_arr = np.asarray(option_type)
    discount_factor = np.exp(-r * T)
    time_decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    call_theta = time_decay - r * K * discount_factor * norm.cdf(d2)
    put_theta = time_decay + r * K * discount_factor * norm.cdf(-d2)
    return np.where(
        option_type_arr == "C",
        call_theta,
        np.where(option_type_arr == "P", put_theta, np.nan),
    )


def calc_spot_vega(S, T, sigma, K, r):
    d1 = calc_spot_d1(S, T, sigma, K, r)
    return S * np.sqrt(T) * norm.pdf(d1)
