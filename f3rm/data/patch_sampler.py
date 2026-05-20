import torch
from torch import Tensor
from typing import Optional, Union
from nerfstudio.data.pixel_samplers import PixelSampler

class PatchGridPixelSampler(PixelSampler):
    def sample_method(
        self,
        batch_size: int,
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Tensor:
        # Match the MaskCLIP grid for 720p: 24 rows, 42 columns
        grid_h, grid_w = 24, 42
        
        # 1. Calculate boundaries
        y_coords = torch.linspace(0, image_height - 1, grid_h + 1, device=device)
        x_coords = torch.linspace(0, image_width - 1, grid_w + 1, device=device)
        
        # 2. Get midpoints (exact patch centers)
        y_centers = (y_coords[:-1] + y_coords[1:]) / 2
        x_centers = (x_coords[:-1] + x_coords[1:]) / 2
        
        # 3. Create the rectangular grid
        grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing="ij")
        
        # 4. Select ONE random image and repeat for all 1008 rays (24 * 42)
        img_index = torch.randint(0, num_images, (1,), device=device).repeat(grid_h * grid_w)
        
        # Return as [1008, 3] -> (image_idx, y, x)
        indices = torch.stack([
            img_index, 
            grid_y.reshape(-1), 
            grid_x.reshape(-1)
        ], dim=-1).long()
        
        return indices

    def forward(self, batch_size: int, **kwargs) -> dict:
        print("Forward called")
        image_height = kwargs.get("image_height")
        image_width = kwargs.get("image_width")
        num_images = kwargs.get("num_images")
        device = kwargs.get("device", "cpu")

        # Get the [1008, 3] indices
        indices = self.sample_method(
            batch_size, num_images, image_height, image_width, device=device
        )

        # Calculate the actual pixel area of one patch
        # (1280 / 42) * (720 / 24) = approx 30 * 30 = 900 sq pixels
        patch_width = image_width / 42
        patch_height = image_height / 24
        area_value = patch_width * patch_height

        pixel_area = torch.full(
            (indices.shape[0], 1), 
            area_value, 
            dtype=torch.float32, 
            device=device
        )

        return {
            "indices": indices,       
            "pixel_area": pixel_area  
        }
