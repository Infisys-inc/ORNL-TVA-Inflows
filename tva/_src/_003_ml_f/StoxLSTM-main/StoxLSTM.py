__all__ = ['StoxLSTM']

import torch
import torch.nn as nn
from StoxLSTM_backbone import StoxLSTM_backbone, StoxLSTM_backbone_WO_PCI
from StoxLSTM_layers import series_decomp
from typing import Dict, Union

class Model(nn.Module):
    def __init__(self, hyperparameter: Dict[str, Union[int, float, str, dict]]):
        super().__init__()
        
        self.decomposition = hyperparameter["decomposition"]
        patch_and_CI = hyperparameter["patch_and_CI"]
        look_back_length, prediction_length = hyperparameter["seq_length"], hyperparameter["prediction_length"]
        revin, subtract_last = hyperparameter["revin"], hyperparameter["subtract_last"]
        kernel_size = hyperparameter["kernel_size"]
        d_seq, d_model, d_latent = hyperparameter["d_seq"], hyperparameter["d_model"], hyperparameter["d_latent"]
        patch_size, patch_stride = hyperparameter["patch_size"], hyperparameter["patch_stride"]
        dropout = hyperparameter["drop_out_rate"]
        device = hyperparameter["device"]
        

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size=kernel_size)
            if patch_and_CI:
                self.model_trend = StoxLSTM_backbone(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length, 
                    patch_size=patch_size, patch_stride=patch_stride,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_p_hidden_dim=hyperparameter["mlp_x_p_hidden_dim"], 
                    mlp_x_p_hidden_layers_num=hyperparameter["mlp_x_p_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, 
                    revin=revin, dropout=dropout
                )
                
                self.model_res = StoxLSTM_backbone(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length, 
                    patch_size=patch_size, patch_stride=patch_stride,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_p_hidden_dim=hyperparameter["mlp_x_p_hidden_dim"], 
                    mlp_x_p_hidden_layers_num=hyperparameter["mlp_x_p_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, 
                    revin=revin, dropout=dropout
                )

            else:
                #print('StoxLSTM without patching and channel independence.')
                self.model_trend = StoxLSTM_backbone_WO_PCI(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, revin=revin, dropout=dropout
                )
                
                self.model_res = StoxLSTM_backbone_WO_PCI(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, revin=revin, dropout=dropout
                )

        else:
            if patch_and_CI:
                self.model = StoxLSTM_backbone(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length, 
                    patch_size=patch_size, patch_stride=patch_stride,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_p_hidden_dim=hyperparameter["mlp_x_p_hidden_dim"], 
                    mlp_x_p_hidden_layers_num=hyperparameter["mlp_x_p_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, revin=revin, dropout=dropout
                )
            else:
                print('StoxLSTM without patching and channel independence.')
                self.model = StoxLSTM_backbone_WO_PCI(
                    d_seq=d_seq, d_model=d_model, d_latent=d_latent, 
                    look_back_length=look_back_length, prediction_length=prediction_length,
                    xlstm_h_num_block=hyperparameter["xlstm_h_num_block"], 
                    slstm_h_at=hyperparameter["slstm_h_at"], 
                    xlstm_g_num_block=hyperparameter["xlstm_g_num_block"], 
                    slstm_g_at=hyperparameter["slstm_g_at"],
                    embed_hidden_dim=hyperparameter["embed_hidden_dim"], 
                    embed_hidden_layers_num=hyperparameter["embed_hidden_layers_num"],
                    mlp_z_hidden_dim=hyperparameter["mlp_z_hidden_dim"], 
                    mlp_z_hidden_layers_num=hyperparameter["mlp_z_hidden_layers_num"],
                    mlp_proj_down_hidden_dim=hyperparameter["mlp_proj_down_hidden_dim"], 
                    mlp_proj_down_hidden_layers_num=hyperparameter["mlp_proj_down_hidden_layers_num"],
                    mlp_x_hidden_dim=hyperparameter["mlp_x_hidden_dim"], 
                    mlp_x_hidden_layers_num=hyperparameter["mlp_x_hidden_layers_num"],
                    device=device, subtract_last=subtract_last, revin=revin, dropout=dropout
                )

            
    def forward(self, x):   #x [bs, seq_len, seq_dim]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)

            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # x: [bs, seq_dim, seq_len]
            
            res, res_meanq, res_logvarq, res_meanp, res_logvarp = self.model_res(res_init) #xT_, meanq, logvarq, meanp, logvarp
            trend, trend_meanq, trend_logvarq, trend_meanp, trend_logvarp = self.model_trend(trend_init)
            
            x = res + trend

            meanq, logvarq = torch.cat((res_meanq, trend_meanq), dim=-1), torch.cat((res_logvarq, trend_logvarq), dim=-1)
            meanp, logvarp = torch.cat((res_meanp, trend_meanp), dim=-1), torch.cat((res_logvarp, trend_logvarp), dim=-1) #[bs, N, d_latent*2]

            x = x.permute(0, 2, 1) # x: [bs, seq_len, seq_dim]
        #     res_init, trend_init = self.decomp_module(x)

        #     res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # x: [bs, seq_dim, seq_len]
            
        #     res_xT_P, res_xC_P, res_xT_P_flatten, res_hT, res_gT_, res, res_meanq, res_logvarq, res_meanp, res_logvarp = self.model_res(res_init) #xT_, meanq, logvarq, meanp, logvarp
        #     trend_xT_P, trend_xC_P, trend_xT_P_flatten, trend_hT, trend_gT_, trend, trend_meanq, trend_logvarq, trend_meanp, trend_logvarp = self.model_trend(trend_init)
            
        #     x = res + trend
        #     xT_P = res_xT_P
        #     xC_P = res_xC_P
        #     xT_P_flatten = res_xT_P_flatten
            
        #     meanq, logvarq = res_meanq, res_logvarq
        #     meanp, logvarp = res_meanp, res_logvarp
        #     hT, gT_ = res_hT, res_gT_
        #     #meanq, logvarq = torch.cat((res_meanq, trend_meanq), dim=-1), torch.cat((res_logvarq, trend_logvarq), dim=-1)
        #    # meanp, logvarp = torch.cat((res_meanp, trend_meanp), dim=-1), torch.cat((res_logvarp, trend_logvarp), dim=-1) #[bs, N, d_latent*2]

        #     x = x.permute(0, 2, 1) # x: [bs, seq_len, seq_dim]
            
        else:
            x = x.permute(0, 2, 1)  # x: [bs, seq_dim, seq_len]
            
            x, meanq, logvarq, meanp, logvarp = self.model(x)
            
            x = x.permute(0, 2, 1) #[bs, seq_len, seq_dim]
            
        #return xT_P, xC_P, xT_P_flatten, hT, gT_, x, meanq, logvarq, meanp, logvarp
        return x, meanq, logvarq, meanp, logvarp