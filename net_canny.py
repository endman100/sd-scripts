import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian


class CannyEdgeDetector(nn.Module):
    def __init__(self, threshold=10.0, vae_dtype=torch.float16, device='cuda'):
        super(CannyEdgeDetector, self).__init__()

        self.threshold = threshold
        self.vae_dtype = vae_dtype
        self.device = device

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size]).astype(np.float16)

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight = torch.nn.Parameter(torch.from_numpy(generated_filters).unsqueeze(0).unsqueeze(0).to(self.vae_dtype), requires_grad=False)
        self.gaussian_filter_horizontal.bias = torch.nn.Parameter(torch.from_numpy(np.array([0.0])).to(self.vae_dtype), requires_grad=False)
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight = torch.nn.Parameter(torch.from_numpy(generated_filters.T).unsqueeze(0).unsqueeze(0).to(self.vae_dtype), requires_grad=False)
        self.gaussian_filter_vertical.bias = torch.nn.Parameter(torch.from_numpy(np.array([0.0])).to(self.vae_dtype), requires_grad=False)

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight = torch.nn.Parameter(torch.from_numpy(sobel_filter).unsqueeze(0).unsqueeze(0).to(self.vae_dtype), requires_grad=False)
        self.sobel_filter_horizontal.bias = torch.nn.Parameter(torch.from_numpy(np.array([0.0])).to(self.vae_dtype), requires_grad=False)
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight = torch.nn.Parameter(torch.from_numpy(sobel_filter.T).unsqueeze(0).unsqueeze(0).to(self.vae_dtype), requires_grad=False)
        self.sobel_filter_vertical.bias = torch.nn.Parameter(torch.from_numpy(np.array([0.0])).to(self.vae_dtype), requires_grad=False)

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight = torch.nn.Parameter(torch.from_numpy(all_filters[:, None, ...]).to(self.vae_dtype), requires_grad=False)
        self.directional_filter.bias = torch.nn.Parameter(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))).to(self.vae_dtype), requires_grad=False)


        self.pixel_range = torch.tensor([range(1048576)], device=self.device, dtype=self.vae_dtype, requires_grad=False)

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)


        grad_x_r = grad_x_r.double()
        grad_y_r = grad_y_r.double()
        grad_x_g = grad_x_g.double()
        grad_y_g = grad_y_g.double()
        grad_x_b = grad_x_b.double()
        grad_y_b = grad_y_b.double()

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)
        grad_mag = grad_mag.to(self.vae_dtype)
        all_filtered = self.directional_filter(grad_mag)

        
        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        indices = (inidices_positive.view(-1) * pixel_count + self.pixel_range).squeeze().long()
        print("Shape of all_filtered before view(-1):", inidices_positive.view(-1).shape, inidices_positive.view(-1).max(), inidices_positive.view(-1).min())
        print("Shape of all_filtered before view(-1):", (inidices_positive.view(-1) * pixel_count).shape, (inidices_positive.view(-1) * pixel_count).max(), (inidices_positive.view(-1) * pixel_count).min())
        print("Shape of all_filtered before view(-1):", indices.shape, indices.max(), indices.min())

        channel_select_filtered_positive = all_filtered.view(-1)[indices].view(1,height,width)

        indices = (inidices_negative.view(-1) * pixel_count + self.pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD
        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        thresholded = thresholded.to(self.vae_dtype)
        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


if __name__ == '__main__':
    CannyEdgeDetector()
