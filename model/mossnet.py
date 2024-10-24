import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from PIL import Image

from dataset.data_loader import BatchedInput
from typing import Dict, Tuple


def return_vis(pred, gt, color='r', cmap=None):
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(pred[:,0], pred[:,1], pred[:,2], c=color, s=2, cmap=cmap)
        ax.scatter(gt[:,0], gt[:,1], gt[:,2], c='g', s=2)

        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('auto')
        ax.axis('tight')
        fig.savefig(tmp_file.name)
        plt.close()
        return Image.open(tmp_file.name)

class MossNet(nn.Module):
    def __init__(self, w_depth_loss: float = 10.0, w_offset_loss: float = 100, w_s_loss: float = 100, polydeg : int = 4):
        super().__init__()
        self.depth_net = DualDecNet(n_channels=5, n_out_a=3, n_out_b=1)
        self.mse = nn.MSELoss()
        self.w_depth_loss = w_depth_loss
        self.w_offset_loss = w_offset_loss
        self.w_s_loss = w_s_loss
        self.eps = 1e-9
        self.M = 10
        self.polydeg = polydeg
        print(f"model robot centerline as {self.polydeg}-th order polynomial")

    def loss(self, m_pts, tip_pts, states, residuals) -> Tuple[Tensor, Dict]:

        output_m, label_m = m_pts
        m_loss = self.w_s_loss * self.mse(output_m, label_m)

        output_tip, label_tip = tip_pts
        tip_loss = self.w_s_loss * self.mse(output_tip, label_tip)

        output_states, label_states = states
        state_loss = self.w_s_loss * self.mse(output_states, label_states)

        residual_loss = self.w_offset_loss * residuals.mean()

        total_loss = m_loss + tip_loss + state_loss + residual_loss
        metas = {
            "total_loss": total_loss.item(),
            "m_loss": m_loss.item(),
            "tip_loss": tip_loss.item(),
            "state_loss": state_loss.item(),
            "residual_loss": residual_loss.item(),
        }
        return total_loss, metas
    
    def forward(self, batched_data: Tensor) -> Tensor:
        return self.depth_net(batched_data)
    
    def train_iter(self, batched_data: BatchedInput) -> Tuple[Tensor, Dict]:     
        self.train()
        batched_output_centerline, batched_output_s, batched_output_w = self.forward(batched_data.uv_rgb_img)

        m_output, tip_output, states_output, m_label, tip_label, states_label, residuals = [], [], [], [], [], [], []
        # split batch
        for i in range(len(batched_output_centerline)):
            output_s_i = batched_output_s[i].squeeze(0).reshape(-1)
            output_w_i = batched_output_w[i].squeeze(0).reshape(-1)
            output_centerline_i = batched_output_centerline[i].permute(1, 2, 0).reshape(-1, 3)
            output_s_i = output_s_i
            mat_a = torch.stack([torch.pow(output_s_i, i) for i in range(0, self.polydeg + 1)], dim=1)
            mat_wa = torch.stack([output_w_i * torch.pow(output_s_i, i) for i in range(0, self.polydeg + 1)], dim=1)
            mat_b = output_centerline_i
            if hasattr(torch.linalg, "solve"):
                poly_w = torch.linalg.solve((mat_a.T @ mat_wa), mat_wa.T @ mat_b)
            else:
                print("Warning: torch.linalg.solve n/a, fall back to torch.inverse")
                poly_w = (torch.inverse(mat_a.T @ mat_wa) @ mat_wa.T) @ mat_b
            M_s = torch.linspace(1 / self.M, 1, self.M).to(batched_output_s.device)
            M_s = torch.stack([torch.pow(M_s, i) for i in range(0, self.polydeg + 1)], dim=1)
            m_output_i = M_s @ poly_w
            m_output.append(m_output_i)
            tip_output.append(m_output_i[-1])

            xyz_from_curve = mat_a @ poly_w
            residual = torch.norm(xyz_from_curve - mat_b, dim=1)
            residuals.append(residual)

            label_m_pts_i = batched_data.state[i]
            indices = (torch.linspace(1 / self.M, 1, self.M) * len(label_m_pts_i)).long() - 1
            label_m_pts_i = label_m_pts_i[indices.to(label_m_pts_i.device)]
            m_label.append(label_m_pts_i)
            tip_label.append(label_m_pts_i[-1])

            M_s = torch.linspace(0, 1, len(batched_data.state[i])).to(batched_output_s.device)
            M_s = torch.stack([torch.pow(M_s, i) for i in range(0, self.polydeg + 1)], dim=1)
            states_output_i = M_s @ poly_w
            states_output.append(states_output_i)
            states_label.append(batched_data.state[i])
            
        
        m_output = torch.cat(m_output)
        tip_output = torch.cat(tip_output)
        m_label = torch.cat(m_label)
        tip_label = torch.cat(tip_label)
        states_output = torch.cat(states_output)
        states_label = torch.cat(states_label)
        residuals = torch.cat(residuals)

        total_loss, metas = self.loss(
                                        (m_output, m_label), 
                                        (tip_output, tip_label),
                                        (states_output, states_label),
                                        residuals)
            
        return total_loss, metas
    
    def eval_iter(self, batched_data: BatchedInput) -> Dict:     
        self.eval()
        batched_output_centerline, batched_output_s, batched_output_w = self.forward(batched_data.uv_rgb_img)

        # fit curve for every frame in a batch
        output_mask, output_s, output_xyz, output_M_pts, output_tip, label_M_pts, label_tip = [], [], [], [], [], [], []
        for i in range(len(batched_output_centerline)):
            output_centerline_i = batched_output_centerline[i].permute(1, 2, 0).reshape(-1, 3)
            output_s_i = batched_output_s.squeeze(1)[i].reshape(-1)
            output_mask.append(batched_output_w.squeeze(1)[i].detach().cpu().numpy())
            output_w_i = batched_output_w.squeeze(1)[i].reshape(-1)
            output_s_i = output_s_i

            output_s.append(output_s_i.detach().cpu().numpy())
            output_xyz.append(output_centerline_i.detach().cpu().numpy())
            mat_a = torch.stack([torch.pow(output_s_i, i) for i in range(0, self.polydeg + 1)], dim=1)
            mat_wa = torch.stack([output_w_i * torch.pow(output_s_i, i) for i in range(0, self.polydeg + 1)], dim=1)
            mat_b = output_centerline_i
            if hasattr(torch.linalg, "solve"):
                poly_w = torch.linalg.solve((mat_a.T @ mat_wa), mat_wa.T @ mat_b)
            else:
                print("Warning: torch.linalg.solve n/a, fall back to torch.inverse")
                poly_w = (torch.inverse(mat_a.T @ mat_wa) @ mat_wa.T) @ mat_b
            M_s = torch.linspace(1 / self.M, 1, self.M).to(batched_output_s.device)
            M_s = torch.stack([torch.pow(M_s, i) for i in range(0, self.polydeg + 1)], dim=1)
            m_output_i = M_s @ poly_w
            m_output_i = m_output_i.detach().cpu().numpy()

            output_M_pts.append(m_output_i)
            output_tip.append(m_output_i[-1])

            label_M_pts_i = batched_data.state[i].detach().cpu().numpy()
            indices = (np.linspace(1 / self.M, 1, self.M) * len(label_M_pts_i)).astype(np.int32) - 1
            label_M_pts_i = label_M_pts_i[indices]
            label_M_pts.append(label_M_pts_i)
            label_tip.append(label_M_pts_i[-1])

        output_M_pts = np.stack(output_M_pts)
        label_M_pts = np.stack(label_M_pts)
        output_tip = np.stack(output_tip)
        label_tip = np.stack(label_tip)
        output_mask = np.stack(output_mask)
        output_s = np.stack(output_s)
        output_xyz = np.stack(output_xyz)

        return {
            "o_M_pts": [output_M_pts],
            "l_M_pts": [label_M_pts],
            "o_tip": [output_tip],
            "l_tip": [label_tip],
            "state": [batched_data.state.detach().cpu()],
            "o_xyz": [output_xyz],
            "o_mask": [output_mask],
            "o_s": [output_s],
        }
    
    def compute_metrics(self, outputs) -> Dict:
        total_losses, centerline_losses, s_losses,  M_errs, tip_errs, counts = 0, 0, 0, 0, 0, 0
        vis_centerline, vis_xyz, vis_mask, vis_arc = [], [], [], []
        x_err, y_err, z_err, M_list = [], [], [], []
        max_errs, vis_failure = [], []
        cm = plt.get_cmap("viridis")
        bar = tqdm(desc="Computing metrics", total=len(outputs["outputs"]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (o_m, l_m, o_tip, l_tip, state, o_s, o_mask, o_xyz) in enumerate(
            zip(
                outputs["o_M_pts"], 
                outputs["l_M_pts"],
                outputs["o_tip"], 
                outputs["l_tip"], 
                outputs["state"],
                outputs["o_s"],
                outputs["o_mask"],
                outputs["o_xyz"],
            )):
            
            # compute M point error, tip error and max error
            temp = np.linalg.norm(o_m - l_m, ord=2, axis=-1)
            M_errs += temp.mean()
            M_list.append(temp.mean())
            tip_errs += np.linalg.norm(o_tip - l_tip, ord=2, axis=-1).mean()
            max_errs.append(np.linalg.norm(o_m - l_m, ord=2, axis=-1).max())
            
            # get x, y, z error
            xyz_error = np.mean(abs(o_m - l_m)[0], axis=0)
            x_err.append(xyz_error[0])
            y_err.append(xyz_error[1])
            z_err.append(xyz_error[2])

            counts += 1

            if i % 100 == 0:
                vis_mask.append(cm(o_mask[0]))
                vis_arc.append(cm(o_s[0].reshape(*o_mask[0].shape)))

            #     # centerline visualization
                vis_centerline_ = return_vis(o_xyz[0], state[0], color=o_s[0], cmap="jet")
                vis_centerline.append(np.asarray(vis_centerline_))

            bar.update(1)
        bar.close()
        
        print("M_error", M_errs / counts)
        print("tip_error", tip_errs / counts)
        print("Max_error", np.asarray(max_errs).max())
        
        max_num = 5
        x_ind = np.argpartition(x_err, -max_num)[-max_num:]
        x_err = np.asarray(x_err)[x_ind]
        print("top x errors", np.sort(x_err))

        y_ind = np.argpartition(y_err, -max_num)[-max_num:]
        y_err = np.asarray(y_err)[y_ind]
        print("top y errors", np.sort(y_err))

        z_ind = np.argpartition(z_err, -max_num)[-max_num:]
        z_err = np.asarray(z_err)[z_ind]
        print("top z errors", np.sort(z_err))

        M_ind = np.argpartition(M_list, -max_num)[-max_num:]
        M_list = np.asarray(M_list)[M_ind]
        print("top M errors", np.sort(M_list))
        
        

        return {
            "loss": total_losses / counts, 
            "centerline_loss": centerline_losses / counts,
            "s_loss": s_losses / counts,
            "M_error": M_errs / counts,
            "tip_error": tip_errs / counts,
            "Max_error": np.asarray(max_errs).max(),
            "vis_centerline": vis_centerline,
            "vis_mask": vis_mask,
            "vis_arc": vis_arc,
        }

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, ks, padding=ks//2, stride=stride),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(),
            nn.Conv2d(outc, outc, ks, padding=ks//2, stride=stride),
            nn.BatchNorm2d(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=stride),
                nn.BatchNorm2d(outc),
            )

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class DualDecNet(nn.Module):

    def __init__(self, n_channels, n_out_a, n_out_b):
        super().__init__()

        self.inc = ResContextBlock(n_channels, 64)
        self.down1 = ResidualBlock(64, 128)
        self.down2 = ResidualBlock(128, 256)
        self.down3 = ResidualBlock(256, 256)
        self.down4 = ResidualBlock(256, 256)
        
        self.up1_a = ResidualBlock(256 // 4 + 256, 256)
        self.up2_a = ResidualBlock(256 // 4 + 256, 256)
        self.up3_a = ResidualBlock(256 // 4 + 128, 128)
        self.up4_a = ResidualBlock(128 // 4 + 64, 64)
        self.outc_a = nn.Conv2d(64, n_out_a, kernel_size=1)

        self.up1_b = ResidualBlock(256 // 4 + 256, 256)
        self.up2_b = ResidualBlock(256 // 4 + 256, 256)
        self.up3_b = ResidualBlock(256 // 4 + 128, 128)
        self.up4_b = ResidualBlock(128 // 4 + 64, 64)
        self.outc_b = nn.Conv2d(64, n_out_b, kernel_size=1)

        self.up1_c = ResidualBlock(256 // 4 + 256, 256)
        self.up2_c = ResidualBlock(256 // 4 + 256, 256)
        self.up3_c = ResidualBlock(256 // 4 + 128, 128)
        self.up4_c = ResidualBlock(128 // 4 + 64, 64)
        self.outc_c = nn.Conv2d(64, 1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.pixshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.inc(x)  # 64, h, w
        
        x2 = self.maxpool(x1)  # 64, h2, w2
        x2 = self.down1(x2)  # 128, h2, w2
        
        x3 = self.maxpool(x2)  # 128, h4, w4
        x3 = self.down2(x3)  # 256, h4, w4

        x4 = self.maxpool(x3)  # 256, h8, w8
        x4 = self.down3(x4)  # 256, h8, w8

        x5 = self.maxpool(x4)  # 256, h16, w16
        x5 = self.down4(x5)  # 256, h16, w16

        x5 = self.pixshuffle(x5)  # 64, h8, w8
        xa = self.up1_a(torch.cat([x5, x4], dim=1))  # 256, h8, w8
        xa = self.pixshuffle(xa)
        xa = self.up2_a(torch.cat([xa, x3], dim=1))  # 256, h4, w4
        xa = self.pixshuffle(xa)
        xa = self.up3_a(torch.cat([xa, x2], dim=1))  # 128, h2, w2
        xa = self.pixshuffle(xa)
        xa = self.up4_a(torch.cat([xa, x1], dim=1))  # 64, h, w
        logits_a = self.outc_a(xa)

        xb = self.up1_b(torch.cat([x5, x4], dim=1))  # 256, h8, w8
        xb = self.pixshuffle(xb)
        xb = self.up2_b(torch.cat([xb, x3], dim=1))  # 256, h4, w4
        xb = self.pixshuffle(xb)
        xb = self.up3_b(torch.cat([xb, x2], dim=1))  # 128, h2, w2
        xb = self.pixshuffle(xb)
        xb = self.up4_b(torch.cat([xb, x1], dim=1))  # 64, h, w
        # logits_b = self.outc_b(xb)
        logits_b = torch.sigmoid(self.outc_b(xb))

        xc = self.up1_c(torch.cat([x5, x4], dim=1))  # 256, h8, w8
        xc = self.pixshuffle(xc)
        xc = self.up2_c(torch.cat([xc, x3], dim=1))  # 256, h4, w4
        xc = self.pixshuffle(xc)
        xc = self.up3_c(torch.cat([xc, x2], dim=1))  # 128, h2, w2
        xc = self.pixshuffle(xc)
        xc = self.up4_c(torch.cat([xc, x1], dim=1))  # 64, h, w
        logits_c = torch.sigmoid(self.outc_c(xc))

        return logits_a, logits_b, logits_c
