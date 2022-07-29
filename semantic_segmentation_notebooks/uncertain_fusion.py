import torch
import numpy as np
import efprob_dc as efprob_dc

class DempsterSchaferCombine(torch.nn.Module):
    '''
    DempsterSchaferCombine will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is shape of [batch*height*width, n_classes]
    '''
    def __init__(self, n_classes):
        super(DempsterSchaferCombine, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2, debug_pixel=0):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        if 4 == alpha1.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alhpa2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
        
        #print ("alpha 1 ", debug_pixel, alpha1[debug_pixel])
        #print ("alpha 2 ", debug_pixel, alpha2[debug_pixel])
        
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.n_classes, 1), b[1].view(-1, 1, self.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        
        #print (alpha_a.shape)
        
        #print ("alpha_a ", debug_pixel, alpha_a[debug_pixel])
        return alpha_a
        
        

class EffectiveProbability(torch.nn.Module):
    '''
    EffectiveProbability used efprob library which can do constructive and
    destructive fusion.
    It fuses mutlinomial distribution.
    Assumption: The input tensor is dirichlet the dirichlet output
    It needs to converted to multinomial distribution.
    The input tesnor is for shape [batch, dim, height, width] 
    It gets converted to [batch*height*width, dim] for calculations.
    Code is very slow because of the map 
    The output is shape of [batch*height*width, n_classes]
    '''
    def __init__(self, confusion_matrix, fusion_type='bayes'):
        super(EffectiveProbability, self).__init__()
        
        assert confusion_matrix.ndim == 2
        
        self.confusion_matrix = confusion_matrix
        
        #Effective probability 
        self.n_classes = confusion_matrix.shape[0]
        prior_labels = []
        current_labels = []
        for i in range(self.n_classes):
            prior_labels.append('P'+str(i))
            current_labels.append('C'+str(i))
        self.prior_labels_dom = efprob_dc.Dom(prior_labels)
        self.current_labels_dom = efprob_dc.Dom(current_labels)
        
        self.array_of_states = []
        for s in self.confusion_matrix:
            self.array_of_states.append(efprob_dc.State(s, self.current_labels_dom))
        self.chan = efprob_dc.chan_from_states(self.array_of_states, self.prior_labels_dom)
  
            
        self.fusion_type = fusion_type
        print ("Initialized fusion type : ", fusion_type)
            
    def single_row_calculation(self, prior, current):
        assert prior.ndim == 1
        assert current.ndim == 1
        assert prior.shape[0] == self.n_classes
        assert current.shape[0] == self.n_classes
        
        prior_dc = efprob_dc.State(prior, self.prior_labels_dom)
        
        try:
            if "bayes" == self.fusion_type:
                posterior = prior_dc / (self.chan << efprob_dc.Predicate(current, self.current_labels_dom))
            elif "dampster" == self.fusion_type:
                posterior = self.chan.inversion(prior_dc) >> efprob_dc.State(current, self.current_labels_dom)
            else:
                print ("select fusion_type as bayes or dampster")
        except:
            print("ERROR Printing prior and current", prior, current)

        return posterior.array
    
        
    def forward(self, prior, current):
        assert (prior.ndim == 4) or (prior.ndim == 2) 
        assert (current.ndim == 4) or (current.ndim == 2) 
        assert prior.shape == current.shape
        assert prior.shape[1] == self.n_classes
        assert current.shape[1] == self.n_classes
        
        if 4 == prior.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            prior = prior.permute(0,2,3,1)
            current = current.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            prior = prior.reshape(-1, self.n_classes) 
            current = current.reshape(-1, self.n_classes) 
        
        
        #Converting dirchlet to probability
        prior = prior/prior.sum(dim=1, keepdim=True)
        current = current/current.sum(dim=1, keepdim=True)
        
        prior = prior.cpu().detach().numpy()
        current = current.cpu().detach().numpy()
     
        posterior = map(self.single_row_calculation, 
                    prior, 
                    current)
       
        return torch.tensor(np.array(list(posterior)))
    

class SumUncertainty(torch.nn.Module):
    '''
    SumUncertainty will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is of ndim = 2 or shape of [batch*height*width, n_classes] 
    '''
    def __init__(self, n_classes):
        super(SumUncertainty, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        if 4 == alpha1.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alhpa2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
            
        return alpha1 + alpha2
    
class MeanUncertainty(torch.nn.Module):
    '''
    MeanUncertainty will combine 2 dirichlet distribution.
    The assumption is that you will get a batch of predictions.
    The output is of ndim = 2 or shape of [batch*height*width, n_classes] 
    '''
    def __init__(self, n_classes):
        super(MeanUncertainty, self).__init__()
        self.n_classes = n_classes
        
    def forward(self, alpha1, alpha2):
        assert (alpha1.ndim == 4) or (alpha1.ndim == 2)
        assert (alpha2.ndim == 4) or (alpha1.ndim == 2)
        assert alpha1.shape == alpha2.shape
        assert alpha1.shape[1] == self.n_classes
        assert alpha2.shape[1] == self.n_classes
        
        if 4 == alpha1.ndim:
            # [batch_size,n_classes, height, width] -> [batch_size, height, width, n_classes]
            alpha1 = alpha1.permute(0,2,3,1)
            alhpa2 = alpha2.permute(0,2,3,1)
            # [batch_size, height, width, n_classes] -> [batch_size*height*width, n_classes]
            alpha1 = alpha1.reshape(-1, self.n_classes) 
            alpha2 = alpha2.reshape(-1, self.n_classes) 
            
        return (alpha1 + alpha2)/2