import argparse
import warnings

import numpy as np
import json
from imageio import imread
from PIL import Image

import torch
import torchvision.transforms as transforms

from utils.device import get_device
from utils.loader import load_decoder, att_based_model, scn_based_model
from utils.token import start_token, end_token, unknown_token, padding_token
from utils.url import is_absolute_path, read_image_from_url
from utils.vizualize import visualize_att


warnings.filterwarnings('ignore')

device = get_device()

def imresize(img, shape):
    return np.array(Image.fromarray(img).resize(shape))

def read_image(image_path):
    r"""Reads an image and captions it with beam search.

    Arguments
        image_path (String or File Object): path to image
    Return
        Tensor : image tensors  (1, 3, 256, 256)
    """

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img).to(device)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)

    return image



need_tag = True#args.type in scn_based_model
need_att = True#args.type in att_based_model

from models.encoders.tagger import EncoderTagger
print('Load tagger checkpoint..')
tagger_checkpoint = torch.load(
    ".\pretrained_dict\id\BEST_checkpoint_tagger_coco_id_5_cap_per_img_5_min_word_freq.pth", map_location=lambda storage, loc: storage)

print('Load tagger encoder...')
encoder_tagger = EncoderTagger()
encoder_tagger.load_state_dict(tagger_checkpoint['model_state_dict'])
encoder_tagger = encoder_tagger.to(device)
encoder_tagger.eval()

print('Load word map..')
# Load word map (word2ix)
with open(".\scn_data\id\coco\WORDMAP_coco_id_5_cap_per_img_5_min_word_freq.json", 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

vocab_size = len(word_map)

# Load word map (word2ix)
with open(".\scn_data\id\coco\TAGMAP_coco_id_5_cap_per_img_5_min_word_freq.json", 'r') as j:
    tag_map = json.load(j)
rev_tag_map = {v: k for k, v in tag_map.items()}  # ix2word

caption_checkpoint = torch.load(
    ".\pretrained_dict\id\BEST_checkpoint_attention_scn_coco_id_5_cap_per_img_5_min_word_freq.pth", map_location=lambda storage, loc: storage)

print('Load caption encoder..')
from models.encoders.caption import EncoderCaption
encoder_caption = EncoderCaption()

encoder_caption.load_state_dict(
    caption_checkpoint['encoder_model_state_dict'])
encoder_caption = encoder_caption.to(device)
encoder_caption.eval()

print('Load caption decoder..')
decoder_caption = load_decoder(
    model_type="attention_scn",
    checkpoint=caption_checkpoint['decoder_model_state_dict'],
    vocab_size=vocab_size)
decoder_caption.eval()

print('=========================')

def process():
    file_img = "file.jpg"
    image = read_image(file_img)
    tags = encoder_tagger(image)
    print('Encoding image...')
    encoder_out = encoder_caption(image)

    if need_tag:
        result = decoder_caption.sample(
            5, word_map, encoder_out, tags)

        tags = np.asarray(tags.flatten().tolist())
        tag_index = np.argsort(tags)[-20:]
        print()
        print('Tags defined : ')
        for idx in tag_index:
            print('{} {}'.format(rev_tag_map[idx], tags[idx]))
        print()
    else:
        result = decoder_caption.sample(5, word_map, encoder_out)

    print('=========================')

    try:
        seq, alphas = result  # for attention-based model
    except:
        seq = result  # for scn only-based model

    sentences = ' '.join([rev_word_map[ind] for ind in seq if ind not in {
        word_map[start_token], word_map[end_token], word_map[padding_token]}])

    print('Sentences : {}'.format(sentences))
    print()
    return sentences
    #if need_att:
    #    alphas = torch.FloatTensor(alphas)
    #    # Visualize caption and attention of best sequence
    #    visualize_att(file_img, seq, alphas, rev_word_map, False)









from flask import Flask
from flask import request
app = Flask(__name__)
@app.route("/process",methods=['POST'])
def hello_world():    
    print('saving to file.jpg...')
    request.files['image'].save("file.jpg")
    
    return process()

print("V 5")
