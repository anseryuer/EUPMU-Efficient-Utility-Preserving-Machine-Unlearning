import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import time
from tqdm import tqdm

def get_feature_dict(model_name):
    if model_name.find('VGG') >= 0:
        conv_fea_dict = {
            'features.0': 'features.0',
            'features.3': 'features.3',
            'features.7': 'features.7',
            'features.10': 'features.10',
            'features.14': 'features.14',
            'features.17': 'features.17',
            'features.20': 'features.20',
            'features.24': 'features.24',
            'features.27': 'features.27',
            'features.30': 'features.30',
            'features.34': 'features.34',
            'features.37': 'features.37',
            'features.40': 'features.40',
        }
        linear_fea_dict = {
            'classifier.0': 'classifier.0',
            'classifier.3': 'classifier.3',
        }
        return conv_fea_dict, linear_fea_dict
    elif model_name.lower().find('resnet') >= 0:
        conv_fea_dict = {
            'conv1': 'conv1',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        linear_fea_dict = {
            'fc': 'fc'   # torchvision ResNet uses 'fc' for the final layer
        }
        return conv_fea_dict, linear_fea_dict
    else:
        raise NotImplementedError(f"Feature dictionary for {model_name} not implemented.")


def compute_svd(model, data_loader, conv_fea_dict, linear_fea_dict, printer=print):
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    for md in model.modules():
        if isinstance(md, torch.nn.Dropout):
            md.training = True

    fea_dict = {val: val for val in {**conv_fea_dict, **linear_fea_dict}.values()}
    
    printer(f"Extracting features from layers: {list(fea_dict.keys())}")
    
    tmp_fea_dict = {**fea_dict}
    features = {}
    if 'input' in tmp_fea_dict:
        features['input'] = []
        tmp_fea_dict.pop('input')

    fea_ext = create_feature_extractor(model, tmp_fea_dict)

    covar, svd = {}, {}
    
    with torch.no_grad():
        for imgs, _ in tqdm(data_loader, desc="Computing SVD"):
            imgs = imgs.to(device)
            
            if 'input' in fea_dict:
                features['input'] = imgs

            feats = fea_ext(imgs)
            for fea_name, fea_val in feats.items():
                features[fea_name] = fea_val.detach()

            for layer, fea_name in conv_fea_dict.items():
                if hasattr(eval(f'model.{layer}'), 'kernel_size'):
                    ks = eval(f'model.{layer}').kernel_size
                    padding = eval(f'model.{layer}').padding
                    f = features[fea_name]
                    patch = F.unfold(f, ks, dilation=1, padding=padding, stride=1)
                    fea_dim = patch.shape[1]
                    patch = patch.permute(0, 2, 1).reshape(-1, fea_dim).double()
                    mat = torch.mm(patch.t(), patch)
                    if layer not in covar:
                        covar[layer] = mat
                    else:
                        covar[layer] += mat

            for layer, fea_name in linear_fea_dict.items():
                f = features[fea_name].double().squeeze()
                if f.dim() > 1:
                    mat = torch.mm(f.t(), f)
                    if layer not in covar:
                        covar[layer] = mat
                    else:
                        covar[layer] += mat
    
    for layer in covar:
        stime = time.time()
        U, S, _ = torch.svd(covar[layer])
        svd[layer] = {'U': U, 'S': torch.sqrt(S)}
        printer(f'Layer: {layer} - SVD time: {time.time() - stime:.04f}s')

    printer(f'Total SVD processing time: {time.time() - start_time:.02f}s')
    return svd
