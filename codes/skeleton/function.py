import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor 
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import pickle as pkl
import numpy as np
import pandas as pd
import glob 
from preprocess_function import FrameProcessWrapper, frameProcess

# from heatran3D_preprocessing import mainWrapper

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            #nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
        
class REfeature3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_tup=()):
        super().__init__()

        self.seq = nn.Sequential(
            BasicBlock(in_channels, out_channels, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(30, 54, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(54,108, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(108, 128, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(128, 128, (1, 2, 2), stride=(1, 2, 2))
        )

    def forward(self, x):
        x = self.seq(x)
        x = x.squeeze(-1)
        x= x.permute(0,3,1,2)
        return x


class Feature3D_bf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_tup=()):
        super().__init__()

        #for 53kp
        self.conv3dResi = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(out_channels, out_channels),

            nn.Conv3d(out_channels, 80, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(80, 80),

            nn.Conv3d(80, 96, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(96, 96),

            nn.Conv3d(96, 128, kernel_tup, stride=(1, 2, 2)),
            BasicBlock(128, 128),

            nn.Conv3d(128, 128, (1, 2, 2), stride=(1, 2, 2)),
        )
        

    def forward(self, x):
        x = self.conv3dResi(x)
        x = x.squeeze(-1)
        x= x.permute(0,3,1,2)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
    

class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, classes=10):
        super().__init__()
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, classes)
        )
   

    def forward(self, x):
        out = self.adv_head(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(#50 322 1163 = 50 64 64
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size),
            #nn.Linear(patch_size*in_channels, emb_size),#for signal
            nn.Linear(patch_size**2*in_channels, emb_size), # -> 48 -> 768
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))#1 1 768
        #self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))#for signal
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size ** 2) + 1, emb_size))#seq_length = img_length


    def forward(self, x:Tensor) ->Tensor:
        # print(x.shape)
        b, _, _, _ = x.shape # 50 3 32 32
        #print(1, x.shape)
        x = self.projection(x) # 50 64 768
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b) #50 1 768
        #print(2, x.shape, self.cls_token.shape, cls_tokens.shape, self.positions.shape)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)#50 65 768
        # position
        #print(3, x.shape, self.cls_token.shape, cls_tokens.shape, self.positions.shape)
        x += self.positions 
        #print(4, x.shape, self.cls_token.shape, cls_tokens.shape, self.positions.shape)
        return x    
'''

'''        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 keypoint=53,
                 in_channels=3,
                 patch_size=15,
                 data_emb_size=50,
                 label_emb_size=10,
                 seq_length = 150,
                 depth=3, 
                 n_classes=9,
                 **kwargs):
        super().__init__(
            #REfeature3D(27, 30, kernel_tup=(1, 4, 4)),
            Feature3D_bf(keypoint, 30, kernel_tup=(1, 4, 4)),#for 27kp
            #Feature3D_bf(53, 96, kernel_tup=(1, 4, 4)),#for 53kp-2
            PatchEmbedding_Linear(in_channels, patch_size, data_emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=data_emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            # ClassificationHead(data_emb_size, n_classes)
        )
        

class heatman_Data(Dataset):
    def __init__(self, heat_dir = '/dataset/KETI/subset10_kp53', npy_dir = '', 
                 label_csv='/dataset/NRC_skeleton/heatmap/compact/', 
                 ceiling=128, mode = 53, **kwargs):
        #sub = pd.read_csv('/home/neuronS1/GEN/HeatMan_Trans/test/KETI/lsh_sub10_label.csv').filename.to_list()

        # heat_npy = glob.glob(filename + "/*.pkl")
        self.heat_npy = pd.read_csv(label_csv).skeleton_path.to_list()
        # self.heat_npy.sort()
        self.mode = mode
        self.heat_dir = heat_dir
        
        self.label = []
        self.data = [] 

        self.ppr = FrameProcessWrapper(cutting='custom',ceiling=ceiling)
        # mw = mainWrapper(filename)

        self.kp27 = [0,1,2,5, 6, 7, 8, 11 ,12, 95, 96, 99, 100, 103, 104, 107, 108, 111, 116, 117, 120, 121, 124, 125, 128, 129, 132]
        self.kp53 = [0,1,2,3,4,5,6,7,8,9,10,
                91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
                112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132]

    def load_batch(self, b_index=0, b_size=30, transform=None, target_transform=None):
        cnt = 0

        if self.label:
            del self.data
            del self.label

        self.label = []
        self.data = []

        for a in self.heat_npy[b_index:]:
            cnt +=1
            if cnt > b_size:
                break
            print('\r Load DATA {} / {}:{}  '.format(cnt, b_size,len(self.heat_npy)), end='')

            id = a.split('/')[-1]
            if int(id[-6:-4]) >9:
                cnt =-1
                continue

            if id[0] == '2':
                self.heat_dir = '/dataset/NRC_skeleton2/heatmap/compact/'
            else:
                self.heat_dir = '/dataset/NRC_skeleton/heatmap/compact/'
        
            try:        
                a = glob.glob(self.heat_dir + id[:-4]+ '.pkl')[0]        
                with open(a, 'rb') as f:
                    data = pkl.load(f)
            except:
                print("fail to load file: ",self.heat_dir + id[:-4]+ '.pkl')
                cnt =-1
                continue
            
            assert data['label']<9, (data['dir'], data['label'], id)
            self.label.append(data['label'])

            if self.mode ==27:
                heatmap = data['heatmap'][:, self.kp27, :, :]
            elif self.mode == 53:
                heatmap = data['heatmap'][:, self.kp53, :, :]

            self.data.append(np.reshape(self.ppr.doPreProc(heatmap),(self.mode, 128, 64, 64)))
            f.close()

        self.transform = transform
        self.target_transform = target_transform

    #custom transfomer
    def myTransformer(self, origin, ceiling):
        orf = origin.shape[0]
        return [origin[int(orf / ceiling * spf),0] for spf in range(ceiling)]
    
    def myTargetTransformer(self, sample):
            #print(sample.shape)   
            # channel h w
            # 109 27 3 - > 3 27 109
            return sample
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform:          
            data = self.myTransformer(data)
        if self.target_transform:
            label = self.myTargetTransformer(label)
        
        return data, label