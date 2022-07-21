import torch
import torch.nn.functional as F


class EpipolarPropagation(torch.nn.Module):
    '''
    EpipolarPropagation propagates a batch tensor based on the depth 
    and camera parameters.
    Parameters:
        fill_empty_with_ones : The value to fill the empty tensor. 
    ASSUMPTION: Works on on square images because of the clamping
    TODO : Find a way of better clamping rather than magic number 511
    ToDo Remove the loop for filling
    TODO : Assert of intrinsic and image height . need to check datagenerator
    
    '''
    def __init__(self, K, Kinv, height, width, fill_empty_with_ones=False):
        super(EpipolarPropagation, self).__init__()
        
        assert K.ndim == 2
        assert Kinv.ndim == 2
        assert height == width #Onlt works for square images, update in dataloader
        
        self.fill_empty_with_ones = fill_empty_with_ones
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device,  )
        
        self.K = torch.Tensor(K).float().to(self.device)
        self.Kinv = torch.Tensor(Kinv).float().to(self.device)
        self.height = height
        self.width = width
        
        #checking if cmaera intrinsic is alligned with the image size
        #assert K[0,2]*2 == height
        #commeinting htis as intrinsix image size is 511 and actual image is 512 need to debug
        
        #Getting index of each pixel of image in [n,3, height*width] shape device=self.device
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(height, device=self.device), 
                                        torch.arange(width, device=self.device), indexing='ij')
        self.grid_x = torch.flatten(self.grid_x, start_dim=0)
        self.grid_y = torch.flatten(self.grid_y, start_dim=0)
        ## Stacking to make the matrix 3 x points
        self.index = torch.vstack((self.grid_x, self.grid_y, torch.ones_like(self.grid_y))).float()
        
    def forward(self, image, depth, T, R):
        assert torch.is_tensor(image) == True
        assert torch.is_tensor(depth) == True
        assert torch.is_tensor(T) == True
        assert torch.is_tensor(R) == True
        assert image.ndim == 4
        assert depth.ndim == 3
                
        #Getting depth and flatenning it and making it [n, 1, height*width] shape
        flattened_depth = torch.flatten(depth, start_dim=1)
        flattened_depth = torch.unsqueeze(flattened_depth, dim=1).float().to(self.device)

        #Getting Transformation and Rotation from previous frame
        T = T.float().to(self.device)
        R = R.float().to(self.device)
        

        #Epipolar Geometry
        transposed_index = ( torch.matmul(torch.matmul(torch.matmul(self.K, R), self.Kinv), self.index) + 
                             torch.matmul(self.K,torch.div(T, flattened_depth)) )
        ## Dividing Last column for each image
        transposed_index = torch.div(transposed_index, transposed_index[:,2,:].unsqueeze(1))
        ## Clamping index ,  imagesize subtracting 1
        transposed_index = torch.clamp(transposed_index, min=0, max=self.height-1).long() # 
        #Projecting pixels from previous frame based on the transposed index
        if self.fill_empty_with_ones:
            projected_tensor = torch.ones_like(image, device=self.device)
        else:
            projected_tensor = torch.zeros_like(image, device=self.device)

        image = image.to(self.device)
        #ToDo can this be done without for loop
        for i,p in enumerate(transposed_index):
            projected_tensor[i,:,p[0],p[1]] = image[i,:,self.grid_x, self.grid_y]

        projected_tensor = projected_tensor.float()
        #m = torch.nn.MaxPool2d(2)
        projected_tensor = F.max_pool2d(projected_tensor, kernel_size=2)

        projected_tensor = F.interpolate(projected_tensor, scale_factor=2 )
        
        return projected_tensor
    
