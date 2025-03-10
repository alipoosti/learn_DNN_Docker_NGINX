import sys, json
from transformers import ResNetForImageClassification
import torch
from torchvision import transforms
from PIL import Image

def inference(image):
    """Run inference using ResNet50 on an input PIL image

    Parameters:
    image : input PIL image

    Returns:
    prediction result in json dictionary format
    """

    # loading the pretrained model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Processing the PIL image to correct format to input to model

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # get model output in form of logits
    with torch.no_grad():
        logits = model(input_batch).logits

    # find the associated label for prediction
    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()

    # response
    response = model.config.id2label[predicted_label]
    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }

if __name__ == '__main__':
    img_path = sys.argv[1]
    image = Image.open(img_path)
    result = inference(image)
    print(result)