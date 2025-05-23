import numpy as np
import torch 
class hybrid_eos:

    def __init__(self, K, gamma, gamma_th):

        self.K = K
        self.gamma = gamma
        self.gamma_th = gamma_th

    def press_cold_eps_cold__rho(self,rho):
        press_cold = self.K * rho**self.gamma
        eps_cold   = press_cold / ( rho * ( self.gamma - 1 ) )
        return press_cold,eps_cold

    def eps_th__temp(self,temp):
        return torch.maximum(torch.zeros_like(temp), temp/(self.gamma_th-1))

    def press__eps_rho(self,eps,rho):
        press_cold,eps_cold = self.press_cold_eps_cold__rho(rho)
        eps = torch.maximum(eps,eps_cold)

        return press_cold + ( eps - eps_cold ) * rho * (self.gamma_th-1) 
    
    def press__rho_temp(self,rho,temp):
        eps_th = self.eps_th__temp(temp)
        return eps_th * rho * (self.gamma_th - 1)

    def eps_range__rho(self,rho):
        press_cold = self.K * rho**self.gamma
        eps_cold   = press_cold / ( rho * ( self.gamma - 1 ) )
        return eps_cold, 1e05

    def press_eps__temp_rho(self,temp,rho):
        press_cold,eps_cold = self.press_cold_eps_cold__rho(rho)
        temp = torch.maximum(temp,torch.zeros_like(temp))
        eps_th = self.eps_th__temp(temp)
        #eps_th = 0 
        press = press_cold + eps_th * rho * (self.gamma_th-1)
        eps = eps_cold + eps_th
        return press,eps

    def cs2__temp_rho(self,temp,rho):
        press,eps = self.press_eps__temp_rho(temp,rho)
        h = 1 + eps + press/rho
        return self.gamma_th * press / rho / h 
    
    def cs2__eps_rho(self,eps,rho):
        press = self.press__eps_rho(eps,rho)
        h = 1 + eps + press/rho
        return self.gamma_th * press / rho / h 
    
    def rho__temp_press(self,temp,press):
        return press/(1e-10+temp)