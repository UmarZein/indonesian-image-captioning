import torch
from torch import nn
import torch.nn.functional as F

from models.attention import Attention

from utils.token import start_token, end_token
from utils.device import get_device

device = get_device()


class PureAttention(nn.Module):
    r"""Caption Decoder with Attention Networks

    Arguments
        attention_dim (int): size of attention network
        embed_dim (int): embedding size
        decoder_dim (int): size of decoder's RNN
        vocab_size (int): size of vocabulary
        encoder_dim (int, optional): feature size of encoded images
        dropout (float, optional): dropout
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(PureAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        r"""Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        r"""Loads embedding layer with pre-trained embeddings.

        Arguments
            embeddings (torch.Tensor): pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        r"""Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        Arguments
            fine_tune (boolean): Allow fine tuning?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        r"""Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        Returns 
            torch.Tensor: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        r"""Forward propagation.

        Arguments
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoded_caption (torch.Tensor): encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths (torch.Tensor): caption lengths, a tensor of dimension (batch_size, 1)
        Returns
            (Tuple of torch.Tensor): scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :],
                           attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, beam_size, word_map, encoder_out):
        r"""Reads an image and captions it with beam search.

        Arguments
            beam_size (int): number of sequences to consider at each decode-step
            word_map (Dictionary): word map
            encoder_out (torch.Tensor): output of encoder model, tensor of dimension (1, enc_image_size, enc_image_size, encoder_dim)
        Return
            [String] : caption tokens
            Float    : weights for visualization
        """

        k = beam_size
        vocab_size = len(word_map)

        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map[start_token]]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
            device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            # (s, encoder_dim), (s, num_pixels)
            awe, alpha = self.attention(encoder_out, h)

            # (s, enc_image_size, enc_image_size)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)

            # gating scalar, (s, encoder_dim)
            gate = self.sigmoid(self.f_beta(h))
            awe = gate * awe

            h, c = self.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = (top_k_words / vocab_size).long()  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                   dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map[end_token]]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas
