import torch
from torchvision import transforms

def load_model_state(model, state_dict, strict=True):
    try:
        model.load_state_dict(state_dict)
        return True
    except Exception as e:
        #print(e)
        pass
    # try to load on GPU
    try:
        print("Retry loading by moving model to GPU")
        model.cuda()
        model.load_state_dict(state_dict, strict=strict)
        return True
    except Exception as e:
        #print(e)
        pass
    # try to load from parallel module
    try:
        print("Retry by loading parallel model")
        temp_state_dict = state_dict.copy()
        for k, v in state_dict.items():
            temp_state_dict[k.replace('module.', '')] = temp_state_dict.pop(k)
        model.load_state_dict(temp_state_dict, strict=strict)
        return True
    except Exception as e:
        print(e)
        print("Loading Failed")
        return False


def load_torch_model(model, filename, strict=True):
    try:
        saved_state = torch.load(filename)
        ret = load_model_state(model, saved_state, strict=strict)
        if not ret:
            ret = load_model_state(model, saved_state["state_dict"], strict=strict)
        if ret:
            print("model loaded")
        return ret
    except Exception as e:
        print("Couldn't open save file")
        print(e)
        return False

def get_image_decoder():
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    #denorm_image = inv_normalize(input)
    #image = transforms.toPILImage()(denorm_image)
    #image = transforms.ToPILImage()(input)
    #return image
    return transforms.Compose([inv_normalize, transforms.ToPILImage()])

IMAGE_DECODER = None

def decode_image(input_tensor):
    global IMAGE_DECODER
    if not IMAGE_DECODER:
        IMAGE_DECODER = get_image_decoder()
    return IMAGE_DECODER(input_tensor)